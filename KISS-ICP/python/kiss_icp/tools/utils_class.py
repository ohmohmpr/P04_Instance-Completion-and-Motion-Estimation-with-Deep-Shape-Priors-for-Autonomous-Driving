from bbox import  BBox3D
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
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
    iou: BBox3D = field(init=False, repr=False)

    def __post_init__(self):
        r = R.from_matrix(self.rot)
        q8d_xyzw = r.as_quat()
        q8d = np.array([q8d_xyzw[3], q8d_xyzw[0], q8d_xyzw[1], q8d_xyzw[2]])

        self.iou: BBox3D = BBox3D(self.x, self.y, self.z, 
                                         self.length, self.width, 
                                         self.height, q=q8d)

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
    s_pose_ious: Dict[int, BBox3D] = field(repr=False)
    g_pose_ious: Dict[int, BBox3D] = field(repr=False)
    
    # ## attributes to be excluded in __str__:
    # # https://stackoverflow.com/questions/71384165/trouble-creating-defaultdict-in-dataclass
    # # https://stackoverflow.com/questions/68722516/exclude-some-attributes-from-str-representation-of-a-dataclass
    # _g_points: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict), repr=False)
    
    # def get_point_cloud_inside_bbox(self, source_points, pose):

    #     # Create bounding box in global frame
    #     g_bbox = self.g_pose_visuals[self.last_frame]
    #     box3d = self.create_bbox(g_bbox)
        
    #     # Convert source points in sensor frame to global frame
    #     source_points = np.hstack((np.asarray(source_points), np.ones((np.asarray(source_points).shape[0], 1))))
    #     g_source_points = (pose @ source_points.T).T
    #     g_source_points = g_source_points[:, :3]
        
    #     # Create points
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(g_source_points)

    #     # Get index
    #     idx_points = box3d.get_point_indices_within_bounding_box(pcd.points)
    #     g_points = np.asarray(pcd.points)[idx_points, :]
    #     self._g_points[self.last_frame] = g_points
        
        
    # def create_bbox(self, bbox):
    #     center = [bbox.x, bbox.y, bbox.z]
    #     lwh = [bbox.length, bbox.width, bbox.height]
    #     box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)
    #     return box3d
                

        
                
                
@dataclass
class OutputPCD:
    '''
    Get points inside bounding boxes.
    '''
    PCD: np.ndarray #pts_sensor
    g_pose_visuals: BoundingBox3D
    s_pose_visuals: BoundingBox3D
    g_pose: np.ndarray = field(init=False)
    s_pose: np.ndarray = field(init=False)
    pts_global: np.ndarray = field(init=False)
    pts_obj_sensor: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.get_g_pose()
        self.get_s_pose()
        self.get_pts_obj_sensor()
        self.get_pts_global()
        
    def get_g_pose(self):
        pose = np.hstack((self.g_pose_visuals.rot, np.array([[self.g_pose_visuals.x], [self.g_pose_visuals.y], [self.g_pose_visuals.z]])))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        self.g_pose = pose
        
    def get_s_pose(self):
        pose = np.hstack((self.s_pose_visuals.rot, np.array([[self.s_pose_visuals.x], [self.s_pose_visuals.y], [self.s_pose_visuals.z]])))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        self.s_pose = pose
    
    def get_pts_obj_sensor(self):
        inv_pose = np.linalg.inv(self.s_pose)
        
        pcd = np.hstack((self.PCD, np.ones((self.PCD.shape[0] , 1))))

        pcd = (inv_pose @ pcd.T).T
        self.pts_obj_sensor = pcd[:, :3]
    
    def get_pts_global(self):
        pcd = np.hstack((self.pts_obj_sensor, np.ones((self.pts_obj_sensor.shape[0] , 1))))

        pcd = (self.g_pose @ pcd.T).T
        self.pts_global = pcd[:, :3]
        
