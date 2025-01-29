# P04 - Instance Completion and Motion Estimation with Deep Shape Priors for Autonomous Driving


## Prerequisite

This code is developed on ubuntu 20.04 and 22.04

1. anaconda or conda with python 3.8.10

```bash
conda create -n msr_p python=3.8.10
conda activate msr_p
pip3 install -r requirements.txt
pip3 install nuscenes-devkit==1.0.5 # https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md

```

2. Ubuntu with CUDA
[CUDA installation](www.google.com)

## Installtion package.
1. Run OpenPCDet to generate tranformation matrix(RENAME later).
2. Install KISS-ICP

```sh
cd KISS-ICP/python
pip install --verbose .
```

## Development on KISS-ICP version 0.2.10

### DELL_G15 - ohm - CURRENT PLATFORM

working mainly on Argoverse2 datasets

```sh
pip3 uninstall kiss-icp -y && pip3 install --verbose KISS-ICP/python
```


Test on KITTI
```sh
kiss_icp_pipeline --visualize ~/data/kiss-icp/KITTI/00/velodyne results/OpenPCDet_PointRCNN/KITTI/00_01.npy
kiss_icp_pipeline --visualize ~/data/kiss-icp/KITTI/04/velodyne results/OpenPCDet_PointRCNN/KITTI/04_01.npy
kiss_icp_pipeline --visualize ~/data/kiss-icp/KITTI/07/velodyne results/OpenPCDet_PointRCNN/KITTI/07_01.npy
kiss_icp_pipeline --visualize ~/data/kiss-icp/KITTI/20/velodyne results/OpenPCDet_PointRCNN/KITTI/20_01.npy # highway
kiss_icp_pipeline --visualize ~/data/kiss-icp/KITTI/21/velodyne results/OpenPCDet_PointRCNN/KITTI/21_01.npy # highway
```

Test on Argoverse2
<https://argoverse.github.io/user-guide/getting_started.html>
<https://www.argoverse.org/av2.html#download-link>

```bash
bash ../sandbox/argoverse/av2-api/conda/install.sh
```

be able to load track-uuid but we nned to integrate av2-api into this repository
and add groud truth pose.

```sh
conda activate av2

kiss_icp_pipeline --visualize ~/data/datasets/av2/ results/OpenPCDet_PointRCNN/KITTI/21_01.npy --dataloader argoverse2 --sequence 001000 --generate_pcd # 1. extract point cloud first 
kiss_icp_pipeline --visualize ~/data/datasets/av2/ results/OpenPCDet_PointRCNN/Argoverse2/000/000026.npy --dataloader argoverse2 --sequence 000026 # 3. show results


# Part 2 in flash drive
kiss_icp_pipeline --visualize /media/ohmpr/OHM_1TB/data/datasets/av2 results/OpenPCDet_PointRCNN/KITTI/21_01.npy --dataloader argoverse2 --sequence 002000 # 1. extract point cloud first 
```

```
# conda environments:
#
base                     /home/ohmpr/anaconda3
av2py3.10                /home/ohmpr/anaconda3/envs/av2py3.10
                         /home/ohmpr/anaconda3/envs/av2py3.10/envs/av2
carlapy3.8               /home/ohmpr/anaconda3/envs/carlapy3.8
deep_sdf_debug           /home/ohmpr/anaconda3/envs/deep_sdf_debug
msr_p                    /home/ohmpr/anaconda3/envs/msr_p

typically you activate /home/ohmpr/anaconda3/envs/av2py3.10/envs/av2
so, if you want to develop further, you need to figure out how to install this environment
and you need to clone av2_api.
another thing i delete results/pcd_argo and results/pcd because extract point cloud and put
in your computer is stupid you need to write a proper code.
good luck to you, the future me.
```
3D detection

```sh
python setup.py develop
conda activate msr_p
cd OpenPCDet/tools/
python3 demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt weight/kitti/pv_rcnn_8369.pth --data_path ../../results/pcd_argo/001/001000/ --ext .npy # 2. run detection


python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../../results/pcd_argo/000012/ --ext .npy # number of col ->> failed
```

Not sure how to get correctly ground truth

### IPB

```sh
pip3 uninstall kiss-icp -y && pip3 install --verbose KISS-ICP/python
```

```sh
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data/kitti/00/velodyne/ ../sequences/sequence00_01.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data/kitti/07/velodyne/ ../sequences/sequence07_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_02.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/20/velodyne/ ../sequences/sequence20_01.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/20/velodyne/ ../sequences/sequence20_02.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/21/velodyne/ ../sequences/sequence21_01.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/21/velodyne/ results/OpenPCDet_PointRCNN/KITTI/21_01.npy # highway
```

### MacOS - ohm - DEPRECATED

```sh
pip3 uninstall kiss-icp -y && pip3 install --verbose KISS-ICP/python
```

```sh
kiss_icp_pipeline --visualize data/kiss-icp/KITTI/00/velodyne results/OpenPCDet_PointRCNN/KITTI/00_01.npy
kiss_icp_pipeline --visualize data/kiss-icp/KITTI/04/velodyne results/OpenPCDet_PointRCNN/KITTI/04_01.npy
kiss_icp_pipeline --visualize data/kiss-icp/KITTI/07/velodyne results/OpenPCDet_PointRCNN/KITTI/07_01.npy

kiss_icp_pipeline --visualize data/kiss-icp/KITTI/20/velodyne results/OpenPCDet_PointRCNN/KITTI/20_01.npy # highway
kiss_icp_pipeline --visualize data/kiss-icp/KITTI/21/velodyne results/OpenPCDet_PointRCNN/KITTI/21_01.npy # highway
```

```sh
python3 DEEP_SDF/reconstruct_object.py
python3 DEEP_SDF/reconstruct_frame1.py
```


python DEEP_SDF/canonical_space.py --config DEEP_SDF/configs/config_kitti.json        

dont forget weight directory              
python3 DEEP_SDF_ss/visualizer_online_multiple_meshes.py                  

refactor code and understand what happen.        
