import numpy as np
import datetime
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, BatchNorm2d as BN2D
from torch_cluster import knn
from torch_geometric.data import Data, DataListLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate

import metric


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def get_dataloader(dataset, atlasize=0, batch_size=32, augment=True):
    xs, ys = [], []
    np.random.seed()
    num_batches = dataset.get_num_batches(batch_size)

    for _ in range(num_batches):
        batch_data, batch_label, batch_weight = dataset.sample_batch_in_all_files(PARAMS["batch_size"], augment)
        for i in range(batch_size):
            xs.append(torch.from_numpy(batch_data[i]).to(torch.float))
            ys.append(torch.from_numpy(batch_label[i]).to(torch.long))
    data_list = []
    atlasend = 3 + atlasize * atlasize * 3
    for (x, y) in zip(xs, ys):
        data = Data(pos=x[:, :3], atlas=x[:, 3:atlasend], x=x[:, atlasend:], y=y)
        data_list.append(data)

    data_loader = DataListLoader(data_list, batch_size)
    return data_loader


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class MLP_ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(MLP_ResBlock, self).__init__()
        self.channels = channels
        self.linearlist = nn.ModuleList()
        for i in range(1, len(self.channels) - 1):
            self.linearlist.append(Seq(Lin(self.channels[i - 1], self.channels[i]), BN(self.channels[i]), ReLU()))
        self.linearlist.append(Seq(Lin(self.channels[-2], self.channels[-1]), BN(self.channels[-1])))
        self.downsample = Seq(Lin(self.channels[0], self.channels[-1]), BN(self.channels[-1]))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        identity = x
        for i in range(len(self.linearlist)):
            seqconv = self.linearlist[i]
            x = seqconv(x)
        x += self.downsample(identity)
        x = self.relu(x)
        return x


class TextureModule(torch.nn.Module):
    def __init__(self, outchannels, atlas_size=16):
        super(TextureModule, self).__init__()
        self.atlas_size = atlas_size
        self.outchannels = outchannels
        self.conv1 = nn.Sequential(  # input shape (N,3,12,12))
            nn.Conv2d(in_channels=3,  # input height
                      out_channels=outchannels[0],  # n_filter
                      kernel_size=3,  # filter size
                      stride=1,  # filter step
                      padding=0,  # con2d出来的图片大小不变
                      ),  # output shape (16,10,10)
            BN2D(outchannels[0]),
            ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 3x3采样，output shape (16,5,5)

        )
        self.conv2 = nn.Sequential(nn.Conv2d(outchannels[0], outchannels[1], 2, 1, 0),  # shape (32,4,4)
                                   BN2D(outchannels[1]),
                                   ReLU(),
                                   nn.MaxPool2d(2))  # shape (32,2,2)
        self.lin1 = torch.nn.Linear(outchannels[1] * 4, outchannels[1])

    def forward(self, x):
        x = x.reshape(x.size(0), 3, self.atlas_size,
                      self.atlas_size)  # [N,3*squre(atlas_size)]=======>[N,3,atlas_size,atlas_size]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.outchannels[1] * 4)  # flat (N, outchannel)
        x = self.lin1(x)
        return x


class PointSAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(PointSAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)

        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class PointSAModuleMsg(torch.nn.Module):
    def __init__(self, ratio, rlist, nsamplelist, channelslist):
        super(PointSAModuleMsg, self).__init__()
        self.ratio = ratio
        self.rlist = rlist
        self.nsamplelist = nsamplelist
        self.convlist = nn.ModuleList()
        for i in range(len(channelslist)):
            self.convlist.append(PointConv(MLP_ResBlock(channelslist[i])))

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        x_new = []
        for i, dil in enumerate(self.rlist):
            N, K = idx.size(-1), self.nsamplelist[i]
            row, col = knn(pos, pos[idx], K * dil, batch, batch[idx]
                           )
            if dil > 1:
                index = torch.randint(K * dil, (N, K), dtype=torch.long,
                                      device=row.device)
                arange = torch.arange(N, dtype=torch.long, device=row.device)
                arange = arange * (K * dil)
                index = (index + arange.view(-1, 1)).view(-1)
                row, col = row[index], col[index]
            edge_index = torch.stack([col, row], dim=0)
            conv = self.convlist[i]
            xi = conv(x, (pos, pos[idx]), edge_index)
            x_new.append(xi)
        pos, batch = pos[idx], batch[idx]
        x_new_contact = torch.cat(x_new, dim=1)
        return x_new_contact, pos, batch


class PointGlobalSAModule(torch.nn.Module):
    def __init__(self, channels):
        super(PointGlobalSAModule, self).__init__()
        self.nn = MLP_ResBlock(channels)

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointFPModule(torch.nn.Module):
    def __init__(self, k, channels):
        super(PointFPModule, self).__init__()
        self.k = k
        self.nn = MLP_ResBlock(channels)

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.nn(x)
        return x, pos_skip, batch_skip


class texure_point_net(torch.nn.Module):
    def __init__(self, num_classes):
        super(texure_point_net, self).__init__()

        self.texconv = TextureModule([16, 32], 12)
        self.sa1_module = PointSAModuleMsg(0.15, [3, 6], [16, 32], [[81 + 3, 64, 64], [81 + 3, 64, 64]])
        self.sa2_module = PointSAModuleMsg(0.15, [3, 6], [16, 32], [[128 + 3, 256, 256], [128 + 3, 256, 256]])
        self.sa3_module = PointGlobalSAModule([512 + 3, 1024, 1024])

        self.fp3_module = PointFPModule(1, [1024 + 512, 512, 512])
        self.fp2_module = PointFPModule(3, [512 + 128, 256, 256])
        self.fp1_module = PointFPModule(3, [256 + 81, 128, 128])

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        tex_out = self.texconv(data.atlas)
        x = torch.cat([tex_out, data.x], dim=1)
        sa0_out = (x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp2_out = self.fp3_module(*sa3_out, *sa2_out)
        fp1_out = self.fp2_module(*fp2_out, *sa1_out)
        fp0_out = self.fp1_module(*fp1_out, *sa0_out)

        x, _, _ = fp0_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def train():
    model.train()
    log_string(str(datetime.now()))
    total_loss = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)
    for data_list in train_loader:
        optimizer.zero_grad()
        out = model(data_list)
        y = torch.cat([data.y for data in data_list]).to(out.device)
        loss = F.nll_loss(out, y, torch.from_numpy(train_label_weight).to(out.device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_val = out.max(dim=1)[1]
        for j in range(len(pred_val)):
            confusion_matrix.increment(y[j].item(), pred_val[j].item())

    log_string("mean loss: %f" % (total_loss / float(len(train_loader))))
    log_string("Overall accuracy : %f" % (confusion_matrix.get_accuracy()))
    log_string("Average IoU : %f" % (confusion_matrix.get_mean_iou()))
    iou_per_class = confusion_matrix.get_per_class_ious()
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, NUM_CLASSES):
        log_string("IoU of %s : %f" % (train_dataset.labels_names[i], iou_per_class[i]))


def test():
    model.eval()
    log_string(str(datetime.now()))
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)
    for data_list in test_loader:
        with torch.no_grad():
            out = model(data_list)
        pred = out.max(dim=1)[1]
        y = torch.cat([data.y for data in data_list]).to(out.device)
        for j in range(len(pred)):
            confusion_matrix.increment(y[j].item(), pred[j].item())

    log_string("Overall accuracy : %f" % (confusion_matrix.get_accuracy()))
    log_string("Average IoU : %f" % (confusion_matrix.get_mean_iou()))
    iou_per_class = confusion_matrix.get_per_class_ious()
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, NUM_CLASSES):
        log_string("IoU of %s : %f" % (train_dataset.labels_names[i], iou_per_class[i]))
    confusion_matrix.print_metrics()
    return confusion_matrix.get_accuracy()

# if __name__ == '__main__':
#     ROOT_DIR = os.path.abspath(".")
#     sys.path.append(ROOT_DIR)
#     WORK_DIR = os.path.join(ROOT_DIR, 'data', 'westcoast')
#     sys.path.append(WORK_DIR)
#
#     # Two global arg collections
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_set", default="train", help="train")
#     parser.add_argument("--config_file", default="westcoast.json", help="config file path")
#
#     FLAGS = parser.parse_args()
#     PARAMS = json.loads(open(osp.join(WORK_DIR, FLAGS.config_file)).read())
#     os.makedirs(osp.join(ROOT_DIR, PARAMS["logdir"]), exist_ok=True)
#     LOG_DIR = osp.join(ROOT_DIR, PARAMS["logdir"])
#     # Import dataset
#     train_dataset = OBJDataset(
#         num_points_per_sample=PARAMS["num_point"],
#         split="train",
#         load_texture=True,
#         creat_texture_atlas=True,
#         texture_atlas_size=PARAMS["atlas_size"],
#         box_size_x=PARAMS["box_size_x"],
#         box_size_y=PARAMS["box_size_y"],
#         features=PARAMS["features"],
#         path=WORK_DIR,
#     )
#     test_dataset = OBJDataset(
#         num_points_per_sample=PARAMS["num_point"],
#         split="validation",
#         load_texture=True,
#         creat_texture_atlas=True,
#         texture_atlas_size=PARAMS["atlas_size"],
#         box_size_x=PARAMS["box_size_x"],
#         box_size_y=PARAMS["box_size_y"],
#         features=PARAMS["features"],
#         path=WORK_DIR,
#     )
#
#     print("finish load data")
#     NUM_CLASSES = train_dataset.num_classes
#
#     LOG_FOUT = open(osp.join(ROOT_DIR, PARAMS["logdir"], "log_train.txt"), "w")
#
#     train_label_weight = train_dataset.get_label_weight()
#     test_label_weight = test_dataset.get_label_weight()
#     print("get train data loader")
#     train_loader = get_dataloader(train_dataset,atlasize=PARAMS["atlas_size"], batch_size=PARAMS["batch_size"], augment=True)
#     print("get validation data loader")
#     test_loader = get_dataloader(test_dataset, atlasize=PARAMS["atlas_size"], batch_size=PARAMS["batch_size"], augment=False)
#
#     model = texure_point_net(train_dataset.num_classes)
#     print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
#     model = DataParallel(model,device_ids=[0,1])
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     best_acc =0.0
#     for epoch in range(1, PARAMS["max_epoch"]):
#         print("in epoch", epoch)
#         print("max_epoch", PARAMS["max_epoch"])
#         log_string("**** EPOCH %03d ****" % (epoch))
#         sys.stdout.flush()
#
#         train()
#         if epoch % 5 == 0:
#             log_string("---- EPOCH %03d EVALUATION ----" % (epoch / 5))
#             acc = test()
#             if acc > best_acc:
#                 best_acc = acc
#                 torch.save(model.state_dict(), osp.join(LOG_DIR, "best_model_epoch_%03d.pt" % (epoch)))
#
#     LOG_FOUT.close()
#     torch.save(model.state_dict(), osp.join(LOG_DIR, "model.pt"))
