A  texture integrating deep learning model for mesh semantic segmentation
=============

Introduction
---------------
The purpose of the project is for semantic segmentation of large scale urban meshes. To make the most of geometry and texture information of meshes, we design a deep learning model on mesh COG point cloud with texure conv. The model firstly applies texture convolution on each face to extract texture features. Then some triangular-face-based features are calculated, such as the 3D coordinates of the center of gravity (COG), the normal and the median RGB. Next, the texture features obtained by the 2D convolution and triangular-face-based features are concatenated and recorded on COG points. Finally, a hierarchical deep network is employed to predict the COG point cloud label to achieve considerable segmentation of the urban scene meshes.

Usage
---------------
### requirements
* numpy
* tensorflow-gpu = 1.12.0
* sklearn
* pytorch3d
* open3d
* torch_geometric
* torch_cluster
* tf_ops
###### Note:The compiling of the tf_ops refer to [Open3D-PointNet2-Semantic3D](https://github.com/isl-org/Open3D-PointNet2-Semantic3D). It seems can only work under tensorfow version lower 1.12
### Dataset
Mesh data can be download from [Figshare](https://doi.org/10.6084/m9.figshare.16681849.v1) and [SUM](https://3d.bk.tudelft.nl/projects/meshannotation).
### Parameters
Create an config file for the dataset,  ./data/XXXdataset/XXXdataset.json to set parameters, e.g. max epoch, batch size, etc. An example is in directory ./data/westcoast/westcoast.json.
###Run
##### 1、Train the model
Run
python train.py
##### 2、Predict
Run
python predict.py
##### 3、caculate area of mesh faces
Run
python caculate_mesh_area.py
##### 4、Interpolate the predict results to original COG point cloud
Run
python interploate_mesh_area.py 
##### 5、Covert the COG  results to thematic mesh in .ply format
Run
python io_mesh_ply.py