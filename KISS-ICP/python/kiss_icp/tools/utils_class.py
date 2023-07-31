from bbox import  BBox3D
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from typing import Dict

@dataclass
class BoundingBox3D:
    '''
    pose in s-frame
    '''
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rot: float

@dataclass
class TempInstance:
    s_pose_visual: BoundingBox3D
    ego_car_pose: np.ndarray
    g_pose_visual: BoundingBox3D = field(init=False)
    s_pose_iou: BBox3D = field(init=False)
    g_pose_iou: BBox3D = field(init=False)

    def get_g_pose(self, ego_car_pose):
        hom_s_pose = np.hstack((self.s_pose_visual.x, self.s_pose_visual.y, self.s_pose_visual.z, 1))
        hom_g_pose = ego_car_pose @ hom_s_pose
        
        g_rot = ego_car_pose[:3, :3] @ self.s_pose_visual.rot
        r = R.from_matrix(g_rot)
        g_q8d_xyzw = r.as_quat()
        g_q8d = np.array([g_q8d_xyzw[3], g_q8d_xyzw[0], g_q8d_xyzw[1], g_q8d_xyzw[2]])
        return hom_g_pose[0:3], g_rot, g_q8d
        
    def __post_init__(self):
        r = R.from_matrix(self.s_pose_visual.rot)
        q8d_xyzw = r.as_quat()
        q8d = np.array([q8d_xyzw[3], q8d_xyzw[0], q8d_xyzw[1], q8d_xyzw[2]])
        g_pose, g_rot, g_q8d = self.get_g_pose(self.ego_car_pose)
        self.g_pose_visual: BoundingBox3D = BoundingBox3D(g_pose[0], g_pose[1], g_pose[2], 
                                                self.s_pose_visual.length, self.s_pose_visual.width, 
                                                self.s_pose_visual.height, g_rot)
        self.s_pose_iou: BBox3D = BBox3D(self.s_pose_visual.x, self.s_pose_visual.y, self.s_pose_visual.z, 
                                         self.s_pose_visual.length, self.s_pose_visual.width, 
                                         self.s_pose_visual.height, q=q8d)
        self.g_pose_iou: BBox3D = BBox3D(g_pose[0], g_pose[1], g_pose[2], 
                                         self.s_pose_visual.length, self.s_pose_visual.width, 
                                         self.s_pose_visual.height, q=g_q8d)
        
@dataclass
class Instance:
    id: int
    last_frame: int
    color_code: np.ndarray

    s_pose_visuals: Dict[int, BoundingBox3D]
    g_pose_visuals: Dict[int, BoundingBox3D]
    s_pose_ious: Dict[int, BBox3D]
    g_pose_ious: Dict[int, BBox3D]
        
        
@dataclass
class OutputPCD:
    '''
    Get points inside bounding boxes.
    '''
    PCD: np.ndarray
    g_pose_visuals: BoundingBox3D
    s_pose_visuals: BoundingBox3D
    g_pose: np.ndarray = field(init=False)
    s_pose: np.ndarray = field(init=False)
    canonical_point: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.get_g_pose()
        self.get_s_pose()
        self.get_canonical_points()
        
        
    def get_g_pose(self):
        pose = np.hstack((self.g_pose_visuals.rot, np.array([[self.g_pose_visuals.x], [self.g_pose_visuals.y], [self.g_pose_visuals.z]])))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        self.g_pose = pose
        
    def get_s_pose(self):
        pose = np.hstack((self.s_pose_visuals.rot, np.array([[self.s_pose_visuals.x], [self.s_pose_visuals.y], [self.s_pose_visuals.z]])))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        self.s_pose = pose
    
    def get_canonical_points(self):
        inv_pose = np.linalg.inv(self.g_pose)
        
        pcd = np.hstack((self.PCD, np.ones((self.PCD.shape[0] , 1))))

        pcd = (inv_pose @ pcd.T).T
        self.canonical_point = pcd[:, :3]
        
