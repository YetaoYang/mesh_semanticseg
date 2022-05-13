import json
import os
import sys

import numpy as np

from py3d_mesh import OBJFileData

if __name__ == "__main__":

    ROOT_DIR = os.path.abspath(".")
    sys.path.append(ROOT_DIR)
    WORK_DIR = os.path.join(ROOT_DIR, 'data', 'westcoast')
    sys.path.append(WORK_DIR)
    hyper_params = json.loads(open(os.path.join(WORK_DIR, "westcoast.json")).read())

    # Directories
    LOG_DIR = os.path.join(ROOT_DIR, hyper_params["logdir"])
    sparse_dir = os.path.join(LOG_DIR, "result", "sparse")
    gt_dir = os.path.join(LOG_DIR, "result", "gt")
    test_dir = os.path.join(WORK_DIR, "test")

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
        objpathprefix = os.path.join(test_dir, file_prefix, file_prefix)
        obj = OBJFileData(objpathprefix, True, 0, 30, 30, False, False, 4)
        dense_points = obj.points
        print("dense_points loaded", flush=True)

        # # Dense face area
        dense_area = obj.features[:, 0].flatten()
        areapath = os.path.join(gt_dir, file_prefix + ".area")
        np.savetxt(areapath, dense_area)
