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

working mainly on nuscene datasets

```sh
pip3 uninstall kiss-icp -y && pip3 install --verbose KISS-ICP/python
```

```txt
scene-0061, Parked truck, construction, intersectio... [18-07-24 03:28:47]   19s, singapore-onenorth, #anns:4622
scene-0103, Many peds right, wait for turning car, ... [18-08-01 19:26:43]   19s, boston-seaport, #anns:2046
scene-0655, Parking lot, parked cars, jaywalker, be... [18-08-27 15:51:32]   20s, boston-seaport, #anns:2332
scene-0553, Wait at intersection, bicycle, large tr... [18-08-28 20:48:16]   20s, boston-seaport, #anns:1950
scene-0757, Arrive at busy intersection, bus, wait ... [18-08-30 19:25:08]   20s, boston-seaport, #anns:592
scene-0796, Scooter, peds on sidewalk, bus, cars, t... [18-10-02 02:52:24]   20s, singapore-queensto, #anns:708
scene-0916, Parking lot, bicycle rack, parked bicyc... [18-10-08 07:37:13]   20s, singapore-queensto, #anns:2387
scene-1077, Night, big street, bus stop, high speed... [18-11-21 11:39:27]   20s, singapore-hollandv, #anns:890
scene-1094, Night, after rain, many peds, PMD, ped ... [18-11-21 11:47:27]   19s, singapore-hollandv, #anns:1762
scene-1100, Night, peds in sidewalk, peds cross cro... [18-11-21 11:49:47]   19s, singapore-hollandv, #anns:935
```

```sh
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/0061-new-new.npy --dataloader nuscenes --sequence 0061 # 
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/0655-new-new.npy --dataloader nuscenes --sequence 0553 # useless
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/0655-new-new.npy --dataloader nuscenes --sequence 0655 # too slow cause many cars.
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/0757-new-new.npy --dataloader nuscenes --sequence 0757 # pretty good
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/0796-new-new.npy --dataloader nuscenes --sequence 0796 # pretty fast and interesting gt at the end of seq.
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/1077-new-new.npy --dataloader nuscenes --sequence 1077 # hard to tell
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/1094-new-new.npy --dataloader nuscenes --sequence 1094 # too many cars and no moving
kiss_icp_pipeline --visualize ~/data/sets/nuscenes/ results/OpenPCDet_PointRCNN/NuScences/1100-new-new.npy --dataloader nuscenes --sequence 1100 # might be interesting at the end of seq.
```

```sh
python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-mini

python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../nuscenes_point/0061_sweep/points0.npy --ext .npy

# cbgs_pp_multihead
python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../nuscenes_point/0061_sweep/points11.npy --ext .npy

# cbgs_pp_multihead - good
python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_second_multihead.yaml --ckpt weight/nuscenes/cbgs_second_multihead_nds6229_updated.pth --data_path ../nuscenes_point/0061_sweep/points38.npy --ext .npy

python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_second_multihead.yaml --ckpt weight/nuscenes/cbgs_second_multihead_nds6229_updated.pth --data_path ../../data/nuscenes_point/0061/points230.npy --ext .npy


```

```txt
https://github.com/open-mmlab/OpenPCDet/issues/257
https://github.com/open-mmlab/OpenPCDet/blob/master/docs/DEMO.md
```

```sh
cd OpenPCDet/tools/
# GROUND TRUTH
python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../nuscenes_point/ --ext .npy

# detected 
python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../../data/nuscenes_point/0061/points0.npy  --ext .npy
python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../../data/nuscenes_point/0757/ --ext .npy

## use kitti for nuscenes
python3 demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt weight/kitti/pv_rcnn_8369.pth --data_path ../../data/nuscenes_point_as_kitti_format/0061/points0.npy --ext .npy
```

use sequence of nuscene to generate ground truth.

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
