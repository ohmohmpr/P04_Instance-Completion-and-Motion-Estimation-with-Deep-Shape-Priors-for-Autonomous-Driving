<p align="center">

  <h1 align="center"> Master Project 04 - Instance Completion and Motion Estimation with Deep Shape Priors for Autonomous Driving</h1>

  <p align="center">
    <!-- <a href="https://github.com/PRBonn/PIN_SLAM/releases"><img src="https://img.shields.io/github/v/release/PRBonn/PIN_SLAM?label=version" /></a> -->
    <!-- <a href="https://github.com/PRBonn/PIN_SLAM#run-pin-slam"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a> -->
    <!-- <a href="https://github.com/PRBonn/PIN_SLAM#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a> -->
    <!-- <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a> -->
    <!-- <a href="https://github.com/PRBonn/PIN_SLAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a> -->
  </p>
  
  <p align="center">
    <a href=""><strong>Panyawat Ohm Rattana</strong></a>
    .
    <a href=""><strong>Shashank</strong></a>
    .
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan</strong></a>
    <!-- <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a> -->
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>

  <h3 align="center">
    <!-- <a href="https://drive.google.com/file/d/1gY25pfbnZ_KeUwcbUAccAKJr72j1Wias/view?usp=sharing">Paper</a> |
    <a href="https://docs.google.com/presentation/d/1rFiMRartneJGU4c9A1rqQXVVotHU9rTvLhfQCqjQ6PQ/edit?usp=sharing">Presentation</a> | -->
  </h3>

  <div align="center">
  </div>
</p>

Examples

Joint Optimization Process
![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/joint_optimization.gif)
Mesh of detected car ground truth car.
![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/mesh_vs_det_vs_gt.gif)
car in a scene
![blue_car_in_a_scene](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/blue_car_in_a_scene.gif)
KITTI dataset
![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/cars_in_kitti.gif)


<!-- **TL;DR: PINI-SLAM is PIN-SLAM with IMU** -->

<!-- ![pin_slam_teaser](https://github.com/PRBonn/PIN_SLAM/assets/34207278/b5ab4c89-cdbe-464e-afbe-eb432b42fccc)

*Globally consistent point-based implicit neural (PIN) map built with PIN-SLAM in Bonn. The high-fidelity mesh can be reconstructed from the neural point map.*

----

![pin_slam_loop_compare](https://github.com/PRBonn/PIN_SLAM/assets/34207278/7dadd438-5a46-451a-9add-c9c08dcae277)

*Comparison of (a) the inconsistent mesh with duplicated structures reconstructed by PIN LiDAR odometry, and (b) the globally consistent mesh reconstructed by PIN-SLAM.*

----

| Globally Consistent Mapping | Various Scenarios | RGB-D SLAM Extension |
| :-: | :-: | :-: |
| <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/b157f24c-0220-4ac4-8cf3-2247aeedfc2e'> | <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/0906f7cd-aebe-4fb7-9ad4-514d089329bd'> | <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/4519f4a8-3f62-42a1-897e-d9feb66bfcd0'> | -->

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <!-- <li>
      <a href="#contribution">Contribution</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run-pini-slam">How to run PINI-SLAM</a>
    </li> -->
    <!-- <li>
    <li>
      <a href="#visualizer-instructions">Visualizer instructions</a>
    </li> -->
    <!-- <li>
      <a href="#results">Results</a>
    </li> -->
    <!-- <li>
      <a href="#citation">Citation</a>
    </li> -->
    <li>
      <a href="#contact">Contact</a>
    </li>
    <!-- <li>
      <a href="#related-projects">Related projects</a>
    </li> -->
  </ol>
</details>

## Abstract

<details>
Accurate pose estimation of surrounding vehicles is crucial for robust autonomous driving. Existing methods often suffer from outliers and inaccuracies, particularly in challenging environments. Inspired by recent object-oriented SLAM approaches that use shape priors, this work presents a novel method for robustly estimating the poses and shapes of surrounding vehicles in challenging autonomous driving scenarios. This work proposes a novel approach that jointly optimizes a single deep shape code and multiple transformation parameters for each vehicle across multiple frames. While relying on a simple IoU-based tracking algorithm to maintain associations. Our multi-frame pose and shape optimization approach leverages the temporal consistency of vehicle shapes and object tracking information and tries to demonstrate the effectiveness of using deep shape priors to improve the reconstruction, detection, and tracking quality of the cars in the scene.
</details>

<!-- ## Contribution

## Installation

### Platform requirement

* Ubuntu OS (tested on 22.04)

* With GPU (recommended) or CPU only (run much slower)

* GPU memory requirement (> 6 GB recommended)

* Windows/MacOS with CPU-only mode

<details>
  <summary>[Details (click to expand)]</summary>

### 1. Set up conda environment

```bash
conda create --name pini python=3.8
conda activate pini
```

### 2. Install the key requirement PyTorch

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

The commands depend on your CUDA version. You may check the instructions [previous-versions](https://pytorch.org/get-started/previous-versions/).

### 3. Install other dependency

```bash
pip3 install -r requirements.txt
```

</details>

## Run PINI-SLAM

### Clone the repository

```bash
git clone https://github.com/ohmohmpr/PINI.git
cd PINI
```

<!-- ### Sanity test

For a sanity test, do the following to download an example part (first 100 frames) of the KITTI dataset (seq 00):

```
sh ./scripts/download_kitti_example.sh
```

And then run:

```
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -vsm
```

<details>
  <summary>[Details (click to expand)]</summary>
  
You can visualize the SLAM process in PIN-SLAM visualizer and check the results in the `./experiments` folder.

Use `run_demo_sem.yaml` if you want to conduct metric-semantic SLAM using semantic segmentation labels:
```
python3 pin_slam.py ./config/lidar_slam/run_demo_sem.yaml -vsm
```

If you are running on a server without an X service (you may first try `export DISPLAY=:0`), then you can turn off the visualization `-v` flag:
```
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -sm
```

If you don't have a Nvidia GPU on your device, then you can turn on the CPU-only operation by adding the `-c` flag:
```
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -vsmc
```

</details> -->

<!-- ### Run on your datasets -->

<!-- For an arbitrary data sequence, you can run with the default config file by:
```
python3 pin_slam.py -i /path/to/your/point/cloud/folder -vsm
``` -->

<!-- <details>
  <summary>[Details (click to expand)]</summary> -->

<!-- Follow the instructions on how to run PIN-SLAM by typing:
```
python3 pin_slam.py -h
```

To run PIN-SLAM with a specific config file, you can run:
```
python3 pin_slam.py path_to_your_config_file.yaml -vsm
```

The flags `-v`, `-s`, `-m` toggle the visualizer, map saving and mesh saving, respectively.

To specify the path to the input point cloud folder, you can either set `pc_path` in the config file or set `-i INPUT_PATH` upon running.

For pose estimation evaluation, you may also set `pose_path` in the config file to specify the path to the reference pose file (in KITTI or TUM format).

For some popular datasets, you can also set the dataset name and sequence name upon running. For example: -->
<!-- Current support datasets.

```bash
# M2DGR
python3 pin_slam.py ./config/lidar_slam/run_m2dgr.yaml rosbag_ohm /velodyne_points -i ~/data/m2dgr/street_03/ -dv

# NTU Viral
python3 pin_slam.py ./config/lidar_slam/run_ntu_viral.yaml rosbag_ohm /os1_cloud_node1/points -i ~/data/NTU_VIRAL/eee_03/ -dv

# Newer College dataset 64 beams
python3 pin_slam.py ./config/lidar_slam/run_newer_college64.yaml rosbag_ohm /os1_cloud_node/points -i ~/data/newer_college_dataset/64/01/bin/0.bag -dv

# Newer College dataset 128 beams
python3 pin_slam.py ./config/lidar_slam/run_newer_college.yaml rosbag_ohm /os_cloud_node/points -i ~/data/ncd/128/collection_1_newer_college/2021-07-01-10-37-38-quad-easy-002.bag -dv

``` -->
<!-- ```
# KITTI dataset sequence 00
python3 pin_slam.py ./config/lidar_slam/run_kitti.yaml kitti 00 -vsm

# MulRAN dataset sequence KAIST01
python3 pin_slam.py ./config/lidar_slam/run_mulran.yaml mulran kaist01 -vsm

# Newer College dataset sequence 01_short
python3 pin_slam.py ./config/lidar_slam/run_ncd.yaml ncd 01 -vsm

# Replica dataset sequence room0
python3 pin_slam.py ./config/rgbd_slam/run_replica.yaml replica room0 -vsm
``` -->
<!-- 
We also support loading data from rosbag, mcap or pcap using specific data loaders (originally from [KISS-ICP](https://github.com/PRBonn/kiss-icp)). You need to set the flag `-d` to use such data loaders. For example:
```
# Run on a rosbag or a folder of rosbags with certain point cloud topic
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag point_cloud_topic_name -i /path/to/the/rosbag -vsmd

# If there's only one topic for point cloud in the rosbag, you can omit it
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag -i /path/to/the/rosbag -vsmd
```

The data loaders for [some specific datasets](https://github.com/PRBonn/PIN_SLAM/tree/main/dataset/dataloaders) are also available. For example, you can run on Replica RGB-D dataset without preprocessing the data by:
```
# Download data
sh scripts/download_replica.sh

# Run PIN-SLAM
python3 pin_slam.py ./config/rgbd_slam/run_replica.yaml replica room0 -i data/Replica -vsmd 
```

The SLAM results and logs will be output in the `output_root` folder set in the config file or specified by the `-o OUTPUT_PATH` flag. 

For evaluation, you may check [here](https://github.com/PRBonn/PIN_SLAM/blob/main/eval/README.md) for the results that can be obtained with this repository on a couple of popular datasets. 

The training logs can be monitored via Weights & Bias online if you set the flag `-w`. If it's your first time using Weights & Bias, you will be requested to register and log in to your wandb account. You can also set the flag `-l` to turn on the log printing in the terminal. -->

<!-- </details> -->

<!-- ### ROS 1 Support

If you are not using PIN-SLAM as a part of a ROS package, you can avoid the catkin stuff and simply run:

```
python3 pin_slam_ros.py path_to_your_config_file.yaml point_cloud_topic_name
```

<details>
  <summary>[Details (click to expand)]</summary>

For example:

```
python3 pin_slam_ros.py ./config/lidar_slam/run.yaml /os_cloud_node/points
```

After playing the ROS bag or launching the sensor you can then visualize the results in Rviz by:

```
rviz -d ./config/pin_slam_ros.rviz 
```

You may use the ROS service `save_results` and `save_mesh` to save the results and mesh (at a default resolution) in the `output_root` folder.

```
rosservice call /pin_slam/save_results
rosservice call /pin_slam/save_mesh
```

The process will stop and the results and logs will be saved in the `output_root` folder if no new messages are received for more than 30 seconds.

If you are running without a powerful GPU, PIN-SLAM may not run at the sensor frame rate. You need to play the rosbag with a lower rate to run PIN-SLAM properly.

You can also put `pin_slam_ros.py` into a ROS package for `rosrun` or `roslaunch`.

We will add support for ROS2 in the near future.

</details>

### Inspect the results after SLAM

After the SLAM process, you can reconstruct mesh from the PIN map within an arbitrary bounding box with an arbitrary resolution by running:

```
python3 vis_pin_map.py path/to/your/result/folder [marching_cubes_resolution_m] [(cropped)_map_file.ply] [output_mesh_file.ply] [mesh_min_nn]
```

<details>
  <summary>[Details (click to expand)]</summary>

The bounding box of `(cropped)_map_file.ply` will be used as the bounding box for mesh reconstruction. This file should be stored in the `map` subfolder of the result folder. You may directly use the original `neural_points.ply` or crop the neural points in software such as CloudCompare. The argument `mesh_min_nn` controls the trade-off between completeness and accuracy. The smaller number (for example `6`) will lead to a more complete mesh with more guessed artifacts. The larger number (for example `15`) will lead to a less complete but more accurate mesh. The reconstructed mesh would be saved as `output_mesh_file.ply` in the `mesh` subfolder of the result folder.

For example, for the case of the sanity test described above, run:

```
python3 vis_pin_map.py ./experiments/sanity_test_*  0.2 neural_points.ply mesh_20cm.ply 8
```
</details> -->

<!-- ## Visualizer Instructions

We provide a PIN-SLAM visualizer based on [lidar-visualizer](https://github.com/PRBonn/lidar-visualizer) to monitor the SLAM process. You can use `-v` flag to turn on it.

<details>
  <summary>[Keyboard callbacks (click to expand)]</summary>

| Button |                                          Function                                          |
|:------:|:------------------------------------------------------------------------------------------:|
|  Space |                                        pause/resume                                        |
| ESC/Q  |                           exit                                                             |
|   G    |                     switch between the global/local map visualization                      |
|   E    |                     switch between the ego/map viewpoint                                   |
|   F    |                     toggle on/off the current point cloud  visualization                   |
|   M    |                         toggle on/off the mesh visualization                               |
|   A    |                 toggle on/off the current frame axis & sensor model visualization          |
|   P    |                 toggle on/off the neural points map visualization                          |
|   D    |               toggle on/off the training data pool visualization                           |
|   I    |               toggle on/off the SDF horizontal slice visualization                         |
|   T    |              toggle on/off PIN SLAM trajectory visualization                               |
|   Y    |              toggle on/off the ground truth trajectory visualization                       |
|   U    |              toggle on/off PIN odometry trajectory visualization                           |
|   R    |                           re-center the view point                                         |
|   Z    |              3D screenshot, save the currently visualized entities in the log folder       |
|   B    |                  toggle on/off back face rendering                                         |
|   W    |                  toggle on/off mesh wireframe                                              |
| Ctrl+9 |                                Set mesh color as normal direction                          |
|   5    |   switch between point cloud for mapping and for registration (with point-wise weight)     |
|   7    |                                      switch between black and white background             |
|   /    |   switch among different neural point color mode, 0: geometric feature, 1: color feature, 2: timestamp, 3: stability, 4: random             |
|  <     |  decrease mesh nearest neighbor threshold (more complete and more artifacts)               |
|  >     |  increase mesh nearest neighbor threshold (less complete but more accurate)                |
|  \[/\] |  decrease/increase mesh marching cubes voxel size                                          |
|  ↑/↓   |  move up/down the horizontal SDF slice                                                     |
|  +/-   |                  increase/decrease point size                                              |

</details> -->

<!-- ## Results

We provide a PIN-SLAM visualizer based on [lidar-visualizer](https://github.com/PRBonn/lidar-visualizer) to monitor the SLAM process. You can use `-v` flag to turn on it.

<details>
  <summary>NTU VIRAL [click to expand]</summary>

| Method       |      PIN-SLAM     |  PINI(Ours) | Video | PIN(add ground)| Vertical LiDAR|
|:------------:|:-----------------:|:-----------:|:-----:|:------:|:------:|
| type         | neural point [m]  | neural point [m]|   |        |        |
|nya_01(easy)  |   **0.117**       |    0.154    |[video nya_01_10x](https://github.com/ohmohmpr/PINI/blob/main/video/nya_01_10x.gif)|        |        |
|nya_02(easy)  |   **0.151**       |    0.221    |       |        |        |
|nya_03(easy)  |   0.305           |    **0.271**|       |        |        |
|tnp_01(easy)  |   0.174           |    **0.144**|       |        |        |
|tnp_02(easy)  |   0.729           |    **0.127**|[video tnp_02_13x](https://github.com/ohmohmpr/PINI/blob/main/video/tnp_02_13x.gif)|        |        |
|tnp_03(easy)  |   **0.161**       |    0.189    |       |        |        |
|eee_01(medium)|     x             |    **0.235**|[video eee_01_14x](https://github.com/ohmohmpr/PINI/blob/main/video/eee_01_14x.gif)|        |        |
|eee_02(medium)|   0.611           |    **0.168**|       |        |        |
|eee_03(medium)|   0.564           |    **0.229**|       |        |        |
|sbs_01(medium)|     x             |    **0.182**|       |        |        |
|sbs_02(medium)|   1.017           |    **0.241**|[video sbs_02_10x](https://github.com/ohmohmpr/PINI/blob/main/video/sbs_02_10x.gif)|[video sbs_02_pin](https://github.com/ohmohmpr/PINI/blob/main/video/sbs_02_pin.gif)|        |
|sbs_03(medium)|     x             |    **0.185**|       |        |        |
|rtp_01(medium)|     x             |       x     |       |        |        |
|rtp_02(medium)|     x             |    **0.322**|       |        |        |
|rtp_03(medium)|   **0.319**       |    0.497    |[video rtp_03_16x](https://github.com/ohmohmpr/PINI/blob/main/video/rtp_03_16x.gif)|        |        |
|spms_01(hard) |     x             |       x     |       |        |        |
|spms_02(hard) |     x             |       x     |       |        |        |
|spms_03(hard) |     x             |       x     |[video spms_03_hort_4x](https://github.com/ohmohmpr/PINI/blob/main/video/spms_03_hort_4x.gif)|        |[video spms_03_vert_2x](https://github.com/ohmohmpr/PINI/blob/main/video/spms_03_vert_2x.gif)|

</details>

<details>
  <summary>Newer College 128beams [click to expand]</summary>

| Method       |      PIN-SLAM     |  PINI(Ours) | Video | PIN|
|:------------:|:-----------------:|:-----------:|:-----:|:------:|
| type         | neural point [m]  | neural point [m]|   |        |
| 3_math_easy  |                   |             |       |        |
| 3_math_medium|                   |             |[video quad_mid_pini_weight_6x](https://github.com/ohmohmpr/PINI/blob/main/video/quad_mid_pini_weight_6x.gif)|[video quad_mid_pin_6x](https://github.com/ohmohmpr/PINI/blob/main/video/quad_mid_pin_6x.gif)|
| 3_math_hard  |                   |             |       |        |

</details>

<details>
  <summary>Bias [click to expand]</summary>

| Method       |      PIN-SLAM     |  PINI(Ours) | LIO-EKF | fail|
|:------------:|:-----------------:|:-----------:|:-----:|:------:|
| type         | neural point [m]  | neural point [m]|   |        |
| cloister_128s|[cloister_128s_pin_5x.gif](https://github.com/ohmohmpr/PINI/blob/main/video/cloister_128s_pin_5x.gif)|[cloister_128s_pini_4x.gif](https://github.com/ohmohmpr/PINI/blob/main/video/cloister_128s_pini_4x.gif)|[cloister_128s_lio_ekf_5x.gif](https://github.com/ohmohmpr/PINI/blob/main/video/cloister_128s_lio_ekf_5x.gif)|[cloister_pini_full_1_5x.gif](https://github.com/ohmohmpr/PINI/blob/main/video/cloister_pini_full_1_5x.gif)|

</details> -->

<!-- ## Citation

If you use PIN-SLAM for any academic work, please cite our original [paper](https://ieeexplore.ieee.org/document/10582536).

```
@article{pan2024tro,
author = {Y. Pan and X. Zhong and L. Wiesmann and T. Posewsky and J. Behley and C. Stachniss},
title = {{PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency}},
journal = IEEE Transactions on Robotics (TRO),
year = {2024},
codeurl = {https://github.com/PRBonn/PIN_SLAM},
}
``` -->

## Contact

If you have any questions, please contact:

* Panyawat Rattana {[panyawat.rattana@hotmail.com](panyawat.rattana@hotmail.com)}

## Related Projects

[KISS-ICP (RAL 23)](https://github.com/PRBonn/kiss-icp): A LiDAR odometry pipeline that just works

---- -->
