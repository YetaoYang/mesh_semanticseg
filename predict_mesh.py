import argparse
import json
import os
import sys
import time

import numpy as np
import open3d
import torch
from torch_geometric.nn import DataParallel, fps

from py3d_mesh import OBJDataset

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)
WORK_DIR = os.path.join(ROOT_DIR, 'data', 'westcoast')
sys.path.append(WORK_DIR)

import metric
from torch_geometric.data import Data, DataListLoader
from texture_point_net import texure_point_net

# Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_samples",
    type=int,
    default=3000,
    help="# samples, each contains num_point points_centered",
)
parser.add_argument("--pt", default="", help="Checkpoint file")
parser.add_argument("--set", default="test", help="train, validation, test")
flags = parser.parse_args()
hyper_params = json.loads(open(os.path.join(WORK_DIR, "westcoast.json")).read())

# Create output dir
LOG_DIR = os.path.join(ROOT_DIR, hyper_params["logdir"])
output_dir = os.path.join(LOG_DIR, "result", "sparse")
os.makedirs(output_dir, exist_ok=True)

outgt_dir = os.path.join(LOG_DIR, "result", "gt")
os.makedirs(outgt_dir, exist_ok=True)

# Dataset
dataset = OBJDataset(
    num_points_per_sample=hyper_params["num_point"],
    split=flags.set,
    load_texture=True,
    creat_texture_atlas=True,
    texture_atlas_size=hyper_params["atlas_size"],
    box_size_x=hyper_params["box_size_x"],
    box_size_y=hyper_params["box_size_y"],
    features=hyper_params["features"],
    path=WORK_DIR,
    combinefiles=False,
)

# Model
NUM_CLASSES = dataset.num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = texure_point_net(NUM_CLASSES)
model = DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load(os.path.join(LOG_DIR, "best_model_epoch_005.pt"), map_location="cuda"))
model.to(device)
batch_size = 128
num_batches = 3
confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

for semantic_file_data in dataset.list_file_data:
    print("Processing {}".format(semantic_file_data))
    ##first write mesh data to ground truth as .pcd and .label
    file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)
    pcdgt = open3d.geometry.PointCloud()
    pcdgt.points = open3d.utility.Vector3dVector(semantic_file_data.points)
    pcdgt_path = os.path.join(outgt_dir, file_prefix + ".pcd")
    open3d.io.write_point_cloud(pcdgt_path, pcdgt)
    print("Exported gt pcd to {}".format(pcdgt_path))

    gt_labels = semantic_file_data.labels
    gt_labels_path = os.path.join(outgt_dir, file_prefix + ".labels")
    np.savetxt(gt_labels_path, gt_labels, fmt="%d")

    # Predict for num_samples times
    points_collector = []
    pd_labels_collector = []
    # prob_collector = []

    # If flags.num_samples < batch_size, will predict one batch
    xs = []
    ys = []
    data_list = []

    ratio = (num_batches * batch_size) / len(semantic_file_data.points)
    points = torch.from_numpy(semantic_file_data.points).to(torch.float)
    idx = fps(points, ratio=ratio, random_start=True)
    centerpoints = points[idx].numpy()
    np.random.shuffle(centerpoints)
    for i in range(num_batches):
        samplepoints = centerpoints[i * batch_size: (i + 1) * batch_size]
        batch_data, batch_label, batch_raw = semantic_file_data.split_sample_batch(batch_size,
                                                                                   dataset.num_points_per_sample,
                                                                                   samplepoints)
        points_collector.extend(batch_raw)
        for j in range(batch_size):
            xs.append(torch.from_numpy(batch_data[j]).to(torch.float))
            ys.append(torch.from_numpy(batch_label[j]).to(torch.long))

    atlasize = int(hyper_params["atlas_size"])
    atlasend = 3 + atlasize * atlasize * 3
    for (x, y) in zip(xs, ys):
        data = Data(pos=x[:, :3], atlas=x[:, 3:atlasend], x=x[:, atlasend:], y=y)
        data_list.append(data)
    data_loader = DataListLoader(data_list, batch_size)
    cm = metric.ConfusionMatrix(NUM_CLASSES)
    for data_list in data_loader:
        s = time.time()
        with torch.no_grad():
            out = model(data_list)
        pd_labels = out.max(dim=1)[1]
        print(
            "Batch size: {}, time: {}".format(batch_size, time.time() - s)
        )

        # Save to collector for file output
        pd_labels_collector.extend(pd_labels)
        # prob_collector.extend(np.array(out.cpu()))

        # Increment confusion matrix
        y = torch.cat([data.y for data in data_list]).to(out.device)
        for j in range(len(pd_labels)):
            confusion_matrix.increment(y[j].item(), pd_labels[j].item())
            cm.increment(y[j].item(), pd_labels[j].item())

    # Save sparse point cloud and predicted labels
    sparse_points = np.array(points_collector).reshape((-1, 3))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(sparse_points)
    pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
    open3d.io.write_point_cloud(pcd_path, pcd)
    print("Exported sparse pcd to {}".format(pcd_path))

    sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
    pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
    np.savetxt(pd_labels_path, sparse_labels, fmt="%d")
    print("Exported sparse labels to {}".format(pd_labels_path))
    cm.print_metrics()
confusion_matrix.print_metrics()
