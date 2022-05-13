import os

import numpy as np
import torch
from pytorch3d.io import load_obj
from torch_cluster import fps

from point_cloud_util import load_features, rotate_feature_point_cloud, rotate_point_cloud, load_labels

train_file_prefixes = [
    "Tile_1",
    "Tile_3",
    "Tile_6",
    "Tile_8",
    "Tile_11",
    "Tile_12",
    "Tile_14",
    "Tile_15",
    "Tile_19",
    "Tile_23",
    "Tile_26",
    "Tile_28",
    "Tile_30",
    "Tile_31",
    "Tile_34",
    "Tile_37",
    "Tile_39",
    "Tile_41",
    "Tile_45",
    "Tile_47",
    "Tile_48",
    "Tile_49",
    "Tile_50",
    "Tile_52",
]

validation_file_prefixes = [
    "Tile_2",
    "Tile_9",
    "Tile_18",
    "Tile_21",
    "Tile_25",
    "Tile_27",
    "Tile_42",
    "Tile_46",
]

test_file_prefixes = [
    "Tile_4",
    # "Tile_5",
    # "Tile_7",
    # "Tile_10",
    # "Tile_13",
    # "Tile_16",
    # "Tile_17",
    # "Tile_20",
    # "Tile_22",
    # "Tile_24",
    # "Tile_29",
    # "Tile_32",
    # "Tile_33",
    # "Tile_35",
    # "Tile_36",
    # "Tile_38",
    # "Tile_40",
    # "Tile_43",
    # "Tile_44",
    # "Tile_51",
    # "Tile_53",
    # "Tile_54",
]
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
all_file_prefixes = train_file_prefixes + validation_file_prefixes + test_file_prefixes

map_name_to_file_prefixes = {
    "train": train_file_prefixes,
    "train_full": train_file_prefixes + validation_file_prefixes,
    "validation": validation_file_prefixes,
    "test": test_file_prefixes,
    "all": all_file_prefixes,
}


def compute_face_norm_area(verts_faces):
    face_norm = np.cross(verts_faces[:, 1] - verts_faces[:, 0], verts_faces[:, 2] - verts_faces[:, 0])
    face_area = np.sqrt((face_norm ** 2).sum(axis=1))
    face_area[face_area < 1e-6] = 1e-6
    face_norm /= face_area[:, np.newaxis]
    face_area *= 0.5
    return face_norm, face_area


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


class OBJFileData:
    def __init__(
            self, file_path_without_ext, has_label, features, box_size_x, box_size_y,
            load_texture: bool = True, creat_texture_atlas: bool = True, texture_atlas_size: int = 4,
    ):
        """
        Loads file data
        """
        self.file_path_without_ext = file_path_without_ext
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.texture_atlas_size = texture_atlas_size

        # Load obj
        verts, faces, aux = load_obj(file_path_without_ext + ".obj", load_texture, creat_texture_atlas,
                                     texture_atlas_size)

        ##cogs
        verts_faces = verts[faces.verts_idx]  ###[F,3,3]
        cogs = torch.sum(verts_faces, 1) / 3  ## [F,3]
        self.points = cogs.numpy()

        ##atlas
        if load_texture:
            atlas = (aux.texture_atlas.numpy() * 255).astype(np.int16)  # [F,texture_atlas_size,texture_atlas_size,3]
            atlas = np.transpose(atlas, (0, 3, 1, 2))  # [F,3,texture_atlas_size,texture_atlas_size]
            atlas = atlas.reshape(-1, 3,
                                  texture_atlas_size * texture_atlas_size)  # [F,3*texture_atlas_size*texture_atlas_size]
            median_RGB_features = np.median(atlas, 2)  # [F,3]
            average_RGB_features = np.average(atlas, 2)  # [F,3,]
            standard_RGB_features = np.std(atlas, 2)  # [F,3]
            hist_RGB_features = np.apply_along_axis(lambda a: np.histogram(a, bins=12, range=(0, 255), density=True)[0],
                                                    2, atlas)  # [F,3,12]
            hist_RGB_features = hist_RGB_features.reshape((-1, 36))
            atlas = atlas.reshape(-1,
                                  3 * texture_atlas_size * texture_atlas_size)  # [F,3*texture_atlas_size*texture_atlas_size]
            self.atlas = atlas
        else:
            self.atlas = np.zeros_like(self.points)

        ##features
        norms, area = compute_face_norm_area(verts_faces)
        self.features = np.concatenate((area.reshape((-1, 1)), norms), axis=1)
        if load_texture:
            self.features = np.concatenate(
                (self.features, median_RGB_features, average_RGB_features, standard_RGB_features, hist_RGB_features),
                axis=1)
        if features > 0:
            external_features = load_features(file_path_without_ext + ".features")
            self.features = np.concatenate((self.features, external_features), axis=1)

        ##labels
        if has_label:
            self.labels = load_labels(file_path_without_ext + "_cog_label.txt.merge")
        else:
            self.labels = np.zeros(len(self.points)).astype(bool)

        # if haslabel, remove unlabled points
        if has_label:
            non_zero_idx = np.nonzero(self.labels)
            self.points = self.points[non_zero_idx]
            self.labels = self.labels[non_zero_idx]
            self.atlas = self.atlas[non_zero_idx]
            self.features = self.features[non_zero_idx]
        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.atlas = self.atlas[sort_idx]
        self.features = self.features[sort_idx]
        print("finish loading:" + file_path_without_ext)

    def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
        """
        Get down-sample or up-sample mask to sample points to num_points_per_sample
        """
        # TODO: change this to numpy's build-in functions
        # Shuffling or up-sampling if needed
        if len(points) - num_points_per_sample > 0:
            true_array = np.ones(num_points_per_sample, dtype=bool)
            false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < num_points_per_sample:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[:num_points_per_sample]
        return sample_mask

    def _center_box(self, points):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(points, axis=0)
        shift = np.array(
            [
                box_min[0] + self.box_size_x / 2,
                box_min[1] + self.box_size_y / 2,
                box_min[2],
            ]
        )
        points_centered = points - shift
        return points_centered

    def _extract_z_box(self, center_point):
        """
        Crop along z axis (vertical) from the center_point.

        Args:
            center_point: only x and y coordinates will be used
            points: points (n * 3)
            scene_idx: scene index to get the min and max of the whole scene
        """
        # TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
        box_min = center_point - [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]
        box_max = center_point + [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]

        i_min = np.searchsorted(self.points[:, 0], box_min[0])
        i_max = np.searchsorted(self.points[:, 0], box_max[0])
        mask = (
                np.sum(
                    (self.points[i_min:i_max, :] >= box_min)
                    * (self.points[i_min:i_max, :] <= box_max),
                    axis=1,
                )
                == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(self.points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask

    def sample(self, num_points_per_sample, center_point=None):
        points = self.points

        # Pick a point, and crop a z-box around
        if center_point is None:
            center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self._extract_z_box(center_point)
        points = points[scene_extract_mask]
        labels = self.labels[scene_extract_mask]
        atlas = self.atlas[scene_extract_mask]
        features = self.features[scene_extract_mask]
        sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
        points = points[sample_mask]
        labels = labels[sample_mask]
        atlas = atlas[sample_mask]
        features = features[sample_mask]
        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self._center_box(points)

        return points_centered, points, labels, atlas, features

    def sample_batch(self, batch_size, num_points_per_sample):
        """
        TODO: change this to stack instead of extend
        """
        batch_points_centered = []
        batch_points_raw = []
        batch_labels = []
        batch_atlas = []
        batch_features = []

        for _ in range(batch_size):
            points_centered, points_raw, gt_labels, atlas, features = self.sample(
                num_points_per_sample
            )
            batch_points_centered.append(points_centered)
            batch_points_raw.append(points_raw)
            batch_labels.append(gt_labels)
            batch_atlas.append(atlas)
            batch_features.append(features)

        return (
            np.array(batch_points_centered),
            np.array(batch_points_raw),
            np.array(batch_labels),
            np.array(batch_atlas),
            np.array(batch_features)
        )

    def split_sample_batch(self, batch_size, num_points_per_sample, center_points):
        """
                TODO: change this to stack instead of extend
                """
        batch_data = []
        batch_points_raw = []
        batch_label = []

        for i in range(batch_size):
            points_centered, points_raw, gt_labels, atlas, features = self.sample(
                num_points_per_sample, center_points[i]
            )
            data = np.array(points_centered)
            data = np.column_stack((data, atlas))
            data = np.column_stack((data, features))
            batch_data.append(data)
            batch_label.append(gt_labels)
            batch_points_raw.append(points_raw)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_points_raw = np.array(batch_points_raw)
        return batch_data, batch_label, batch_points_raw

    def sample_mesh(self, num_points_per_sample, overlap=1.0):
        """
                TODO: change this to stack instead of extend
                """
        mesh_data = []
        mesh_raw_point = []
        mesh_label = []
        ratio = overlap / num_points_per_sample
        points = torch.from_numpy(self.points).to(torch.float)
        idx = fps(points, ratio=ratio, random_start=True)
        center_points = points[idx].numpy()
        np.random.shuffle(center_points)
        for i in range(len(center_points)):
            points_centered, points_raw, gt_labels, atlas, features = self.sample(
                num_points_per_sample, center_points[i]
            )
            data = np.array(points_centered)
            data = np.column_stack((data, atlas))
            data = np.column_stack((data, features))
            mesh_data.append(data)
            mesh_label.append(gt_labels)
            mesh_raw_point.append(points_raw)

        return mesh_data, mesh_label, mesh_raw_point


class OBJFilesData:
    def __init__(
            self, obj_file_data_list, box_size_x, box_size_y
    ):
        """
        Loads file data
        """
        self.file_path_without_ext = "combin_files"
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.texture_atlas_size = obj_file_data_list[0].texture_atlas_size
        self.points = obj_file_data_list[0].points
        self.labels = obj_file_data_list[0].labels
        self.atlas = obj_file_data_list[0].atlas
        self.features = obj_file_data_list[0].features
        for i in range(1, len(obj_file_data_list)):
            self.points = np.concatenate((self.points, obj_file_data_list[i].points), axis=0)
            self.labels = np.concatenate((self.labels, obj_file_data_list[i].labels), axis=0)
            self.atlas = np.concatenate((self.atlas, obj_file_data_list[i].atlas), axis=0)
            self.features = np.concatenate((self.features, obj_file_data_list[i].features), axis=0)
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.atlas = self.atlas[sort_idx]
        self.features = self.features[sort_idx]

    def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
        """
        Get down-sample or up-sample mask to sample points to num_points_per_sample
        """
        # TODO: change this to numpy's build-in functions
        # Shuffling or up-sampling if needed
        if len(points) - num_points_per_sample > 0:
            true_array = np.ones(num_points_per_sample, dtype=bool)
            false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < num_points_per_sample:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[:num_points_per_sample]
        return sample_mask

    def _center_box(self, points):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(points, axis=0)
        shift = np.array(
            [
                box_min[0] + self.box_size_x / 2,
                box_min[1] + self.box_size_y / 2,
                box_min[2],
            ]
        )
        points_centered = points - shift
        return points_centered

    def _extract_z_box(self, center_point):
        """
        Crop along z axis (vertical) from the center_point.

        Args:
            center_point: only x and y coordinates will be used
            points: points (n * 3)
            scene_idx: scene index to get the min and max of the whole scene
        """
        # TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
        box_min = center_point - [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]
        box_max = center_point + [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]

        i_min = np.searchsorted(self.points[:, 0], box_min[0])
        i_max = np.searchsorted(self.points[:, 0], box_max[0])
        mask = (
                np.sum(
                    (self.points[i_min:i_max, :] >= box_min)
                    * (self.points[i_min:i_max, :] <= box_max),
                    axis=1,
                )
                == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(self.points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask

    def sample(self, num_points_per_sample, center_point=None):
        points = self.points

        # Pick a point, and crop a z-box around
        if center_point is None:
            center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self._extract_z_box(center_point)
        points = points[scene_extract_mask]
        labels = self.labels[scene_extract_mask]
        atlas = self.atlas[scene_extract_mask]
        features = self.features[scene_extract_mask]
        sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
        points = points[sample_mask]
        labels = labels[sample_mask]
        atlas = atlas[sample_mask]
        features = features[sample_mask]
        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self._center_box(points)

        return points_centered, points, labels, atlas, features

    def sample_batch(self, batch_size, num_points_per_sample):
        """
        TODO: change this to stack instead of extend
        """
        batch_points_centered = []
        batch_points_raw = []
        batch_labels = []
        batch_atlas = []
        batch_features = []

        for _ in range(batch_size):
            points_centered, points_raw, gt_labels, atlas, features = self.sample(
                num_points_per_sample
            )
            batch_points_centered.append(points_centered)
            batch_points_raw.append(points_raw)
            batch_labels.append(gt_labels)
            batch_atlas.append(atlas)
            batch_features.append(features)

        return (
            np.array(batch_points_centered),
            np.array(batch_points_raw),
            np.array(batch_labels),
            np.array(batch_atlas),
            np.array(batch_features)
        )

    def split_sample_batch(self, batch_size, num_points_per_sample, center_points, augment):
        """
                TODO: change this to stack instead of extend
                """
        batch_data = []
        batch_points_raw = []
        batch_label = []

        for i in range(batch_size):
            points_centered, points_raw, gt_labels, atlas, features = self.sample(
                num_points_per_sample, center_points[i]
            )
            data = np.array(points_centered)
            data = np.column_stack((data, atlas))
            data = np.column_stack((data, features))
            batch_data.append(data)
            batch_label.append(gt_labels)
            batch_points_raw.append(points_raw)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_points_raw = np.array(batch_points_raw)

        if augment:
            batch_data = rotate_feature_point_cloud(batch_data,
                                                    3 * self.texture_atlas_size * self.texture_atlas_size + 49)  # 49 features
        return batch_data, batch_label, batch_points_raw


def loadobj(file_path_without_ext, haslabel, features, box_size_x, box_size_y, load_texture, creat_texture_atlas,
            texture_atlas_size):
    objdata = OBJFileData(file_path_without_ext, haslabel, features, box_size_x, box_size_y, load_texture,
                          creat_texture_atlas, texture_atlas_size)
    return objdata


class OBJDataset:
    def __init__(
            self, num_points_per_sample, split, load_texture, creat_texture_atlas, texture_atlas_size,
            features, box_size_x, box_size_y, path, combinefiles=True
    ):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size_x (int): Defaults to 10. The size of the extracted cube.
        box_size_y (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.num_points_per_sample = num_points_per_sample
        self.split = split
        self.load_texture = load_texture
        self.creat_texture_atlas = creat_texture_atlas
        self.texture_atlas_size = texture_atlas_size
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.num_classes = 8
        self.path = path
        self.features = features
        self.labels_names = [
            "unlabeled",
            "roof",
            "facade",
            "window",
            "impervious surface",
            "tree",
            "vehicle",
            "low vegetation",
        ]

        # Get file_prefixes
        file_prefixes = map_name_to_file_prefixes[self.split]
        print("Dataset split:", self.split)
        print("Loading file_prefixes:", file_prefixes)

        # Load files
        from multiprocessing import Pool
        results = []
        pool = Pool(processes=25)
        self.list_file_data = []
        for file_prefix in file_prefixes:
            file_path_without_ext = os.path.join(self.path, self.split, file_prefix, file_prefix)
            results.append(pool.apply_async(loadobj, (file_path_without_ext,
                                                      True,  # self.split != "test",
                                                      self.features,
                                                      self.box_size_x,
                                                      self.box_size_y,
                                                      self.load_texture,
                                                      self.creat_texture_atlas,
                                                      self.texture_atlas_size)))
        pool.close()
        pool.join()
        print("Sub-process(es) done.")
        for res in results:
            objdata = res.get()
            self.list_file_data.append(objdata)

        if combinefiles:
            combineobjs = OBJFilesData(self.list_file_data, self.box_size_x, self.box_size_y)
            self.list_file_data = []
            self.list_file_data.append(combineobjs)
        # Pre-compute the probability of picking a scene
        self.num_scenes = len(self.list_file_data)
        self.scene_probas = [
            len(fd.points) / self.get_total_num_points() for fd in self.list_file_data
        ]

        # Pre-compute the points weights if it is a training set
        if self.split == "train":
            # First, compute the histogram of each labels
            label_weights = np.zeros(self.num_classes)
            for labels in [fd.labels for fd in self.list_file_data]:
                tmp, _ = np.histogram(labels, range(self.num_classes + 1))
                label_weights += tmp
            # Then, a heuristic gives the weights
            # 1 / log(1.2 + probability of occurrence)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = 1 / np.log(1.2 + label_weights)
        else:
            self.label_weights = np.zeros(self.num_classes)

    def sample_batch_in_all_files(self, batch_size, augment=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for _ in range(batch_size):
            points, labels, atlas, features, weights = self.sample_in_all_files(True)
            data = np.array(points)
            if self.load_texture:
                data = np.column_stack((data, atlas))
            data = np.column_stack((data, features))
            batch_data.append(data)
            batch_label.append(labels)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)
        if augment:
            if self.texture_atlas_size or self.features > 0:
                batch_data = rotate_feature_point_cloud(batch_data,
                                                        3 * self.texture_atlas_size * self.texture_atlas_size * int(
                                                            self.load_texture)
                                                        + 49)  # 49 features
            else:
                batch_data = rotate_point_cloud(batch_data)
        return batch_data, batch_label, batch_weights

    def sample_in_all_files(self, is_training):
        """
        Returns points and other info within a z - cropped box.
        """
        # Pick a scene, scenes with more points are more likely to be chosen
        scene_index = np.random.choice(
            np.arange(0, len(self.list_file_data)), p=self.scene_probas
        )
        # print(self.list_file_data[scene_index].file_path_without_ext)
        # Sample from the selected scene
        points_centered, points_raw, labels, atlas, features = self.list_file_data[
            scene_index
        ].sample(num_points_per_sample=self.num_points_per_sample)

        if is_training:
            weights = self.label_weights[labels]
            return points_centered, labels, atlas, features, weights
        else:
            return scene_index, points_centered, points_raw, labels, atlas, features

    def get_total_num_points(self):
        list_num_points = [len(fd.points) for fd in self.list_file_data]
        return np.sum(list_num_points)

    def get_num_batches(self, batch_size, overlapsize=1.0):
        return int(
            self.get_total_num_points() * overlapsize / (batch_size * self.num_points_per_sample)
        )

    def get_total_xyz(self):
        return self.list_file_data[0].points  # returen points for combinefile

    def get_file_paths_without_ext(self):
        return [file_data.file_path_without_ext for file_data in self.list_file_data]

    def get_label_weight(self):
        return self.label_weights


if __name__ == '__main__':
    mesh = OBJFileData("/home/yyt/PycharmProjects/mesh_conv/data/westcoast/validation/Tile_9/Tile_9", has_label=False,
                       features=0, box_size_x=10, box_size_y=10, load_texture=True, creat_texture_atlas=True,
                       texture_atlas_size=4)
