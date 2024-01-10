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
kiss_icp_pipeline --visualize ~/data/datasets/av2/ results/OpenPCDet_PointRCNN/Argoverse2/000000.npy --dataloader argoverse2 --sequence 000000
kiss_icp_pipeline --visualize ~/data/datasets/av2/ results/OpenPCDet_PointRCNN/Argoverse2/000013.npy --dataloader argoverse2 --sequence 000013
```

3D detection

```sh
python setup.py develop
conda activate msr_p
cd OpenPCDet/tools/
python3 demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt weight/kitti/pv_rcnn_8369.pth --data_path ../../results/pcd_argo/000012/ --ext .npy
python3 demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt weight/kitti/pv_rcnn_8369.pth --data_path ../../results/pcd_argo/000012/0.npy --ext .npy


python3 demo.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt weight/nuscenes/pp_multihead_nds5823_updated.pth --data_path ../../results/pcd_argo/000012/ --ext .npy # number of col ->> failed
```

|level|results|remark|
|--|--|--|
|most important| THIS ||
|could be important|||
|not important|||

|seq|results|remark|
|--|--|--|
|00| less interesting| no tracking car|
|01|interesting| one or two opposite direction cars, at the end of seq, we can show how bad bbox are|
|02|interesting| one or two opposite direction cars|
|03|less interesting| one moving car but couldn't track over time|
|04| less interesting| one moving car but couldn't track over time, might be interesting|
|05|interesting| lot of cars|
|06| not interesting| lots of car but cannot track over time|
|07|interesting| one or two opposite direction cars, at the end of seq|
|08| not interesting| lots of car but cannot track over time|
|09|interesting| one moving cars, THIS|
|10|interesting| one moving cars, THIS|
|11| less interesting| ego-car parks at an intersection and be able to observe twos orthogonal moving cars|
|12|interesting| lot of oppsosite direction cars but can track for a few frames|
|13|interesting| two or three moving cars|
|14|no moving car||
|15|one leading car| might be one|
|16|one leading car|use this and remove humans and check ground truth, THIS|
|17|less inrestering leading car| opposite direct car|
|18|could be important| cars on the opposite side of the road|
|19|could be important| ego-car tunrs left|
|20|could be important| detected one car moving in opposite direction for 6 frames|
|21|important| one car is turning, could be a good example, THIS|
|22|nothing interesting| |
|23|nothing interesting| one car turns at an intersection |
|24|nothing interesting| |
|25|nothing interesting| no moving car here|
|26|interesting| this is a good example of how bbox are bad|
|27|could be important| one moving car at the end of seq|
|28|could be important| detection fails on the car|
|29|important| one moving car at the end of seq|
|30|important| one moving car and turning|
|31|interesting | lot of cars, THIS |
|32|could be important | lot of cars, detection fails on the car |
|33|important | lot of cars |
|34|could be important | lot of cars |
|35|important | one or two cars moving in opposite direction |
|36|could be important | because all cars all are standing still |
|37|could be important  | but deteciton failed |
|38| important | parking cars for this is a good example of how bbox are bad |
|39| important | This is the best one. THIS |
|40| not important | no cars in this scene |
|41| could be important | parking at intersection  a good example of how bbox are bad|
|42| could be important  | a good example of how bbox are bad |
|43| important, not sure  | like silom road |
|44| could be important  | detected car at the end of seq |
|45| important  | lots of cars of detected cars, kiss-icp or gt might fail here |
|46| important  | lots of cars, something fail here because gt does not perfectly alging with point cloud |
|47| important  | lots of cars but they are parking cars |
|48| important  | no moving cars in the first half adn moving in the second half, 3D detection couldn't detect the turning cars. |
|49| very important  | lots of cars |

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
