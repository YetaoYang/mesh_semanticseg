import json
import os
import sys

import numpy as np
from pytorch3d.io import load_obj

color_map = [
    [255, 255, 255],
    [255, 0, 51],
    [255, 255, 102],
    [102, 253, 204],
    [192, 192, 192],
    [0, 153, 51],
    [255, 51, 204],
    [127, 255, 0],
]


def save_ply(verts, faces, label, filename):
    vertsize = np.full((len(faces.verts_idx), 1), 3, dtype=int)
    label = label.reshape((-1, 1))
    color = np.array(color_map)[label]
    color = np.squeeze(color)
    faceprop = np.concatenate((vertsize, faces.verts_idx, label, color), axis=1)
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex " + str(len(verts)) + "\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("element face " + str(len(faces.verts_idx)) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property int label\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        np.savetxt(f, verts, fmt="%0.2f")
        f.write("\n")
        np.savetxt(f, faceprop, fmt="%d")
        f.close()


if __name__ == "__main__":

    ROOT_DIR = os.path.abspath(os.path.pardir)
    sys.path.append(ROOT_DIR)
    WORK_DIR = os.path.join(ROOT_DIR, 'data', 'sum')
    sys.path.append(WORK_DIR)
    hyper_params = json.loads(open(os.path.join(WORK_DIR, "sum.json")).read())

    # Directories
    LOG_DIR = os.path.join(ROOT_DIR, hyper_params["logdir"])
    label_dir = os.path.join(LOG_DIR, "result", "dense")
    gt_label_dir = os.path.join(LOG_DIR, "result", "gt")

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
        print("Interpolating_mesh_PLY:", file_prefix, flush=True)

        dense_labels_path = os.path.join(label_dir, file_prefix + ".labels")  # forwestcoast
        dense_labels = np.loadtxt(dense_labels_path, dtype=float).astype(np.int32).flatten()

        gt_labels_path = os.path.join(gt_label_dir, file_prefix + ".labels")
        gt_labels = np.loadtxt(gt_labels_path, dtype=float).astype(np.int32).flatten()

        dense_labels[dense_labels > 8] = gt_labels[dense_labels > 8]
        dense_labels[dense_labels == 0] = gt_labels[dense_labels == 0]
        objpath = os.path.join(WORK_DIR, "test", file_prefix, file_prefix + ".obj")  # forwestcoast
        plypath = os.path.join(LOG_DIR, "result", "dense", file_prefix + ".ply")
        verts, faces, aux = load_obj(objpath, False, False, 0)
        save_ply(verts, faces, dense_labels, plypath)
