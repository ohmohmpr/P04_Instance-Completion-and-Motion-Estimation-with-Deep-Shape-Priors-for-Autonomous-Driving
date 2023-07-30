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
    
    is_added: bool
    
    s_line_set: ... = field(init=False)
    g_line_set: ... = field(init=False)
        
    def __post_init__(self):
        self.s_line_set = self.s_translate_boxes_to_open3d_instance()
        self.g_line_set = self.g_translate_boxes_to_open3d_instance()
        
    def update_line_sets(self):
        self.s_line_set = self.s_update_translate_boxes_to_open3d_instance()
        self.g_line_set = self.g_update_translate_boxes_to_open3d_instance()
        # np.asarray(current_instance.s_line_set.lines))
        
    def s_translate_boxes_to_open3d_instance(self):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        https://github.com/open-mmlab/OpenPCDet/blob/master/tools/visual_utils/open3d_vis_utils.py
        """
        center = [self.s_pose_visuals[self.last_frame].x, self.s_pose_visuals[self.last_frame].y, self.s_pose_visuals[self.last_frame].z]
        lwh = [self.s_pose_visuals[self.last_frame].length, self.s_pose_visuals[self.last_frame].width, self.s_pose_visuals[self.last_frame].height]
        box3d = o3d.geometry.OrientedBoundingBox(center, self.s_pose_visuals[self.last_frame].rot, lwh)

        self.s_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        lines = np.asarray(self.s_line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        self.s_line_set.lines = o3d.utility.Vector2iVector(lines)
        
        return self.s_line_set

    def g_translate_boxes_to_open3d_instance(self):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        https://github.com/open-mmlab/OpenPCDet/blob/master/tools/visual_utils/open3d_vis_utils.py
        """
        center = [self.g_pose_visuals[self.last_frame].x, self.g_pose_visuals[self.last_frame].y, self.g_pose_visuals[self.last_frame].z]
        lwh = [self.g_pose_visuals[self.last_frame].length, self.g_pose_visuals[self.last_frame].width, self.g_pose_visuals[self.last_frame].height]
        box3d = o3d.geometry.OrientedBoundingBox(center, self.g_pose_visuals[self.last_frame].rot, lwh)

        self.g_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        lines = np.asarray(self.g_line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        self.g_line_set.lines = o3d.utility.Vector2iVector(lines)
        return self.g_line_set
        
    def s_update_translate_boxes_to_open3d_instance(self):
        center = [self.s_pose_visuals[self.last_frame].x, self.s_pose_visuals[self.last_frame].y, self.s_pose_visuals[self.last_frame].z]
        lwh = [self.s_pose_visuals[self.last_frame].length, self.s_pose_visuals[self.last_frame].width, self.s_pose_visuals[self.last_frame].height]
        box3d = o3d.geometry.OrientedBoundingBox(center, self.s_pose_visuals[self.last_frame].rot, lwh)
        
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        self.s_line_set.lines = o3d.utility.Vector2iVector(lines)
        return self.s_line_set
        
    def g_update_translate_boxes_to_open3d_instance(self):
        center = [self.g_pose_visuals[self.last_frame].x, self.g_pose_visuals[self.last_frame].y, self.g_pose_visuals[self.last_frame].z]
        lwh = [self.g_pose_visuals[self.last_frame].length, self.g_pose_visuals[self.last_frame].width, self.g_pose_visuals[self.last_frame].height]
        box3d = o3d.geometry.OrientedBoundingBox(center, self.g_pose_visuals[self.last_frame].rot, lwh)
        
        lines = np.asarray(self.g_line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        self.g_line_set.lines = o3d.utility.Vector2iVector(lines)
        return self.g_line_set
        
        
        