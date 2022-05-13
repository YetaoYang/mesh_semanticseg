import json
import os
import sys
import time
from pprint import pprint

import numpy as np
import open3d
import tensorflow as tf

import metric
import point_cloud_util
from tf_ops.tf_interpolate import interpolate_label_with_color


class Interpolator:
    def __init__(self):
        pl_sparse_points = tf.placeholder(tf.float32, (None, 3))
        pl_sparse_labels = tf.placeholder(tf.int32, (None,))
        pl_dense_points = tf.placeholder(tf.float32, (None, 3))
        pl_knn = tf.placeholder(tf.int32, ())
        pl_radius = tf.placeholder(tf.float32, ())
        dense_labels, dense_colors = interpolate_label_with_color(
            pl_sparse_points, pl_sparse_labels, pl_dense_points, pl_knn, pl_radius
        )
        self.ops = {
            "pl_sparse_points": pl_sparse_points,
            "pl_sparse_labels": pl_sparse_labels,
            "pl_dense_points": pl_dense_points,
            "pl_knn": pl_knn,
            "pl_radius": pl_radius,
            "dense_labels": dense_labels,
            "dense_colors": dense_colors,
        }
        self.sess = tf.Session()

    def interpolate_labels(self, sparse_points, sparse_labels, dense_points, knn=3, radius=0.1):
        return self.sess.run(
            [self.ops["dense_labels"], self.ops["dense_colors"]],
            feed_dict={
                self.ops["pl_sparse_points"]: sparse_points,
                self.ops["pl_sparse_labels"]: sparse_labels,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: knn,
                self.ops["pl_radius"]: radius,
            },
        )


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


if __name__ == "__main__":

    ROOT_DIR = os.path.abspath(".")
    sys.path.append(ROOT_DIR)
    WORK_DIR = os.path.join(ROOT_DIR, 'data', 'westcoast')
    sys.path.append(WORK_DIR)
    hyper_params = json.loads(open(os.path.join(WORK_DIR, "westcoast.json")).read())

    # Directories
    LOG_DIR = os.path.join(ROOT_DIR, hyper_params["logdir"])
    sparse_dir = os.path.join(LOG_DIR, "result", "sparse")
    dense_dir = os.path.join(LOG_DIR, "result", "dense")
    gt_dir = os.path.join(LOG_DIR, "result", "gt")
    os.makedirs(dense_dir, exist_ok=True)

    # Parameters
    radius = 0.05
    k = 10

    labels_names = [
        "unlabeled",
        "roof",
        "facade",
        "window",
        "impervious surface",
        "tree",
        "vehicle",
        "low vegetation",
    ]

    # Global statistics
    NUM_CLASSES = 8
    cm_global = metric.ConfusionMatrix(NUM_CLASSES)
    interpolator = Interpolator()
    file_prefixs = [
        "Tile_4",
        "Tile_5",
        "Tile_7",
        "Tile_10",
        "Tile_13",
        "Tile_16",
        "Tile_17",
        "Tile_20",
        "Tile_22",
        "Tile_24",
        "Tile_29",
        "Tile_32",
        "Tile_33",
        "Tile_35",
    ]

    for file_prefix in file_prefixs:
        print("Interpolating:", file_prefix, flush=True)

        # Paths
        sparse_points_path = os.path.join(sparse_dir, file_prefix + ".pcd")
        sparse_labels_path = os.path.join(sparse_dir, file_prefix + ".labels")
        dense_points_path = os.path.join(gt_dir, file_prefix + ".pcd")
        dense_labels_path = os.path.join(dense_dir, file_prefix + ".labels")
        dense_points_colored_path = os.path.join(
            dense_dir, file_prefix + "_colored.pcd"
        )
        dense_gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")
        dense_points_array_path = os.path.join(dense_dir, file_prefix + ".txt")
        dense_area_path = os.path.join(gt_dir, file_prefix + ".area")
        # Sparse points
        sparse_pcd = open3d.io.read_point_cloud(sparse_points_path)
        sparse_points = np.asarray(sparse_pcd.points)
        del sparse_pcd
        print("sparse_points loaded", flush=True)

        # Sparse labels
        sparse_labels = point_cloud_util.load_labels(sparse_labels_path)
        print("sparse_labels loaded", flush=True)

        # Dense points
        dense_pcd = open3d.io.read_point_cloud(dense_points_path)
        dense_points = np.asarray(dense_pcd.points)
        print("dense_points loaded", flush=True)

        # Dense Ground-truth labels
        try:
            dense_gt_labels = point_cloud_util.load_labels(os.path.join(gt_dir, file_prefix + ".labels"))
            print("dense_gt_labels loaded", flush=True)
        except:
            print("dense_gt_labels not found, treat as test set")
            dense_gt_labels = None

        # Assign labels
        start = time.time()
        dense_labels, dense_colors = interpolator.interpolate_labels(
            sparse_points, sparse_labels, dense_points, k, radius
        )
        print("KNN interpolation time: ", time.time() - start, "seconds", flush=True)

        # Write dense labels
        point_cloud_util.write_labels(dense_labels_path, dense_labels)
        print("Dense labels written to:", dense_labels_path, flush=True)

        # Write dense point cloud with color
        dense_pcd.colors = open3d.utility.Vector3dVector(dense_colors)
        open3d.io.write_point_cloud(dense_points_colored_path, dense_pcd)
        dense_labels_2d = np.reshape(dense_labels, (-1, 1))
        dense_array = np.concatenate((dense_points, dense_colors, dense_labels_2d), axis=1)
        np.savetxt(dense_points_array_path, dense_array, fmt="%1.6f %1.6f %1.6f %d %d %d %d")
        print("Dense pcd with color written to:", dense_points_colored_path, flush=True)

        # Eval
        if dense_gt_labels is not None:
            dense_area = np.loadtxt(dense_area_path)
            cm = metric.ConfusionMatrix(NUM_CLASSES)
            for i in range(len(dense_area)):
                cm.increment_plus(dense_gt_labels[i], dense_labels[i], dense_area[i] * 10000)
                cm_global.increment_plus(dense_gt_labels[i], dense_labels[i], dense_area[i] * 10000)
            cm.print_metrics()
    pprint("Global results")
    cm_global.print_metrics()
    LOG_FOUT = open(os.path.join(LOG_DIR, "evaluation.txt"), "w")
    log_string("Overall accuracy : %f" % (cm_global.get_accuracy()))
    log_string("Average IoU : %f" % (cm_global.get_mean_iou()))
    iou_per_class = cm_global.get_per_class_ious()
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, NUM_CLASSES):
        log_string("IoU of %s : %f" % (labels_names[i], iou_per_class[i]))
    LOG_FOUT.close()
