# P04 - Instance Completion and Motion Estimation with Deep Shape Priors for Autonomous Driving


## Abstract
Accurate pose estimation of surrounding vehicles is crucial for robust autonomous driving. Existing methods often suffer from outliers and inaccuracies, particularly in challenging environments. Inspired by recent object-oriented SLAM approaches that use shape priors, this work presents a novel method for robustly estimating the poses and shapes of surrounding vehicles in challenging autonomous driving scenarios. This work proposes a novel approach that jointly optimizes a single deep shape code and multiple transformation parameters for each vehicle across multiple frames. While relying on a simple IoU-based tracking algorithm to maintain associations. Our multi-frame pose and shape optimization approach leverages the temporal consistency of vehicle shapes and object tracking information and tries to demonstrate the effectiveness of using deep shape priors to improve the reconstruction, detection, and tracking quality of the cars in the scene.

## Result

![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/joint_optimization.gif)

![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/mesh_vs_det_vs_gt.gif)

![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/blue_car_in_a_scene.gif)


![](https://github.com/ohmohmpr/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/blob/main/img/cars_in_kitti.gif)
