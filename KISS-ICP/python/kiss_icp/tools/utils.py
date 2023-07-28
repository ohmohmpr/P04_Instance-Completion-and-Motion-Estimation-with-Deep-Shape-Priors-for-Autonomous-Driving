from bbox import  BBox3D
from bbox.metrics import iou_3d
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from typing import Dict

@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    axis_angles: float
    q8d: float = field(init=False)
    bBox3D: BBox3D = field(init=False)
    
    def from_Axis_angle_to_Quaternion(self, axis_angles):
        _axis_angles = np.array([0, 0, axis_angles + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(_axis_angles)
        q8d = Quaternion(matrix=rot)
        return q8d
    
    def __post_init__(self):
        self.q8d: str = self.from_Axis_angle_to_Quaternion(self.axis_angles)
        self.bBox3D: BBox3D = BBox3D(self.x, self.y, self.z, self.length, self.width, self.height, q=self.q8d)
    
@dataclass
class Pose:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    axis_angles: float
    q8d: float
    bBox3D: BBox3D = field(init=False)
    
    def from_Axis_angle_to_Quaternion(self, axis_angles):
        _axis_angles = np.array([0, 0, axis_angles + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(_axis_angles)
        q8d = Quaternion(matrix=rot)
        return q8d
    
    def __post_init__(self):
        self.q8d: str = self.from_Axis_angle_to_Quaternion(self.axis_angles)
        self.bBox3D: BBox3D = BBox3D(self.x, self.y, self.z, self.length, self.width, self.height, q=self.q8d)

@dataclass
class Instance:
    id: int
    s_pose: Dict[int, Pose]
    g_pose: Dict[int, Pose]


class InstanceAssociation:
    def __init__(self):
        self.instances = {}
        self.current_instances = {}
        self.ID = 0
        self.newly_added_instances = {}

    def update(self, bBoxes3D, frames_ID):
        
        self.newly_added_instances = {}
        for bBox3D in bBoxes3D:
            
            s_pose = {}
            g_pose = {}
            s_pose[frames_ID] = Pose(bBox3D.x, bBox3D.y, bBox3D.z, 
                                    bBox3D.length, bBox3D.width, bBox3D.height, 
                                    bBox3D.axis_angles, bBox3D.q8d)
            g_pose[frames_ID] = Pose(bBox3D.x, bBox3D.y, bBox3D.z, 
                                    bBox3D.length, bBox3D.width, bBox3D.height, 
                                    bBox3D.axis_angles, bBox3D.q8d)
            
            self.instances[self.ID] = Instance(self.ID, s_pose, g_pose)
            self.newly_added_instances[self.ID] = Instance(self.ID, s_pose, g_pose)
            self.ID = self.ID + 1
        
    def get_current_instances(self, current_frame, num_consecutive_frames=1):
        ## WE CAN USE QUEUE HERE
        starting_frame = current_frame - num_consecutive_frames + 1
        self.add_newly_added_instances()
        self.delete_old_instances(starting_frame)
                    
        return self.current_instances
         
         
    def add_newly_added_instances(self):
        if self.newly_added_instances:
            for id, newly_added_instance in self.newly_added_instances.items():
                try:
                    self.current_instances[id] = newly_added_instance
                except KeyError:
                    pass
        
    def delete_old_instances(self, starting_frame):
        frame = starting_frame - 1
        idx = []
        for id, instance in self.current_instances.items():
            try:
                if instance.g_pose[frame]:
                    idx.append(id)
            except KeyError:
                    pass
        for id in idx: 
            self.current_instances.pop(id)
        
def translate_boxes_to_open3d_instance(bbox):
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
    center = bbox[0:3]
    lwh = bbox[3:6]
    axis_angles = np.array([0, 0, bbox[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d
    
    
# def translate_boxes_to_open3d_instance(self, bbox):
#     """
#             4-------- 6
#         /|         /|
#         5 -------- 3 .
#         | |        | |
#         . 7 -------- 1
#         |/         |/
#         2 -------- 0
#     """
#     center = bbox[0:3]
#     lwh = bbox[3:6]
#     axis_angles = np.array([0, 0, bbox[6] + 1e-10])
#     rot = self.o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
#     box3d = self.o3d.geometry.OrientedBoundingBox(center, rot, lwh)

#     line_set = self.o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

#     # import ipdb; ipdb.set_trace(context=20)
#     lines = np.asarray(line_set.lines)
#     lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

#     line_set.lines = self.o3d.utility.Vector2iVector(lines)
#     return line_set, box3d