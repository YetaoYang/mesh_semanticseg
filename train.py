import argparse
import datetime
import json
import os
import os.path as osp
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataListLoader
from torch_geometric.nn import DataParallel

import metric
from py3d_mesh import OBJDataset
from texture_point_net import texure_point_net


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


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(".")
    sys.path.append(ROOT_DIR)
    WORK_DIR = os.path.join(ROOT_DIR, 'data', 'westcoast')
    sys.path.append(WORK_DIR)

    # Two global arg collections
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", default="train", help="train")
    parser.add_argument("--config_file", default="westcoast.json", help="config file path")

    FLAGS = parser.parse_args()
    PARAMS = json.loads(open(osp.join(WORK_DIR, FLAGS.config_file)).read())
    os.makedirs(osp.join(ROOT_DIR, PARAMS["logdir"]), exist_ok=True)
    LOG_DIR = osp.join(ROOT_DIR, PARAMS["logdir"])
    # Import dataset
    train_dataset = OBJDataset(
        num_points_per_sample=PARAMS["num_point"],
        split="train",
        load_texture=True,
        creat_texture_atlas=True,
        texture_atlas_size=PARAMS["atlas_size"],
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        features=PARAMS["features"],
        path=WORK_DIR,
    )
    test_dataset = OBJDataset(
        num_points_per_sample=PARAMS["num_point"],
        split="validation",
        load_texture=True,
        creat_texture_atlas=True,
        texture_atlas_size=PARAMS["atlas_size"],
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        features=PARAMS["features"],
        path=WORK_DIR,
    )

    print("finish load data")
    NUM_CLASSES = train_dataset.num_classes

    LOG_FOUT = open(osp.join(ROOT_DIR, PARAMS["logdir"], "log_train.txt"), "w")

    train_label_weight = train_dataset.get_label_weight()
    test_label_weight = test_dataset.get_label_weight()
    print("get train data loader")
    train_loader = get_dataloader(train_dataset, atlasize=PARAMS["atlas_size"], batch_size=PARAMS["batch_size"],
                                  augment=True)
    print("get validation data loader")
    test_loader = get_dataloader(test_dataset, atlasize=PARAMS["atlas_size"], batch_size=PARAMS["batch_size"],
                                 augment=False)

    model = texure_point_net(train_dataset.num_classes)
    print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
    model = DataParallel(model, device_ids=[0, 1])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(1, PARAMS["max_epoch"]):
        print("in epoch", epoch)
        print("max_epoch", PARAMS["max_epoch"])
        log_string("**** EPOCH %03d ****" % (epoch))
        sys.stdout.flush()

        train()
        if epoch % 5 == 0:
            log_string("---- EPOCH %03d EVALUATION ----" % (epoch / 5))
            acc = test()
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), osp.join(LOG_DIR, "best_model_epoch_%03d.pt" % (epoch)))

    LOG_FOUT.close()
    torch.save(model.state_dict(), osp.join(LOG_DIR, "model.pt"))
