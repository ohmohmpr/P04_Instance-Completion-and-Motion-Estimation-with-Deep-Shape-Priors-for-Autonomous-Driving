from bbox import  BBox3D
from bbox.metrics import iou_3d
from dataclasses import dataclass, field
from kiss_icp.tools.utils_class import TempInstance, Instance
import numpy as np
import open3d as o3d
from typing import Dict

color_table = np.load('data/kiss-icp/color_table/color_table.npy')

class InstanceAssociation:
    def __init__(self):
        self.instances = {}
        self.current_instances = {}
        self.ID = 0
        self.newly_added_instances = {}
        self.iou_threshold = 0

    def update(self, ego_car_pose, bBoxes3D, frames_ID):
        
        self.newly_added_instances = {}
        find_in = "global_frame"
        for bBox3D in bBoxes3D:
            tmp_box = TempInstance(bBox3D, ego_car_pose)
            is_found = self.find_in_current_instances(tmp_box, frames_ID, find_in)
                    
            if not is_found:
                ################## SKIP READING ##################
                # Create KEYS
                s_pose_visual = { frames_ID: tmp_box.s_pose_visual }
                g_pose_visual = { frames_ID: tmp_box.g_pose_visual }
                s_pose_iou = { frames_ID: tmp_box.s_pose_iou }
                g_pose_iou = { frames_ID: tmp_box.g_pose_iou }
                ################## SKIP READING ##################
                
                # Add to instances and newly_added_instances
                color_code = color_table[self.ID]
                self.instances[self.ID] = Instance(self.ID, frames_ID, color_code, 
                                                   s_pose_visual, g_pose_visual,
                                                   s_pose_iou, g_pose_iou)
                self.newly_added_instances[self.ID] = Instance(self.ID, frames_ID, color_code,
                                                   s_pose_visual, g_pose_visual,
                                                   s_pose_iou, g_pose_iou)
                self.ID = self.ID + 1
        
        
    def find_in_current_instances(self, tmp_box, frames_ID, find_in):
        is_found = False
        
        for id, current_instance in self.current_instances.items():
            if iou_3d(tmp_box.g_pose_iou, current_instance.g_pose_ious[current_instance.last_frame]) or \
            iou_3d(tmp_box.s_pose_iou, current_instance.s_pose_ious[current_instance.last_frame]) \
            > self.iou_threshold :
                self.update_instances(current_instance, id, frames_ID, tmp_box)
                is_found = True
        
        return is_found
        
    def update_instances(self, instance, id, frames_ID, tmp_box):
        instance.s_pose_visuals[frames_ID] =  tmp_box.s_pose_visual
        instance.g_pose_visuals[frames_ID] =  tmp_box.g_pose_visual
        instance.s_pose_ious[frames_ID] =  tmp_box.s_pose_iou
        instance.g_pose_ious[frames_ID] =  tmp_box.g_pose_iou
        
    def get_current_instances(self, current_frame, num_consecutive_frames=1):
        ## WE CAN USE QUEUE HERE
        starting_frame = current_frame - num_consecutive_frames + 1
        self.add_newly_added_instances()
        self.get_only_last_frame_for_each_instance(starting_frame, current_frame)
        self.delete_old_instances(starting_frame)
        
        return self.current_instances
         
    def add_newly_added_instances(self):
        if self.newly_added_instances:
            for id, newly_added_instance in self.newly_added_instances.items():
                try:
                    self.current_instances[id] = newly_added_instance
                except KeyError:
                    pass
                
                
    def get_only_last_frame_for_each_instance(self, starting_frame, current_frame):
        for id, instance in self.current_instances.items():
            for f in range(current_frame, starting_frame-1, -1):
                try:
                    if instance.g_pose_ious[f]:
                        instance.last_frame = f
                        self.pop(instance, f, starting_frame)
                except KeyError:
                        pass
         
    def pop(self, instance, frame, starting_frame):
        if frame == starting_frame:
            return
        try:
            instance.s_pose_visuals.pop(frame-1)
            instance.g_pose_visuals.pop(frame-1)
            instance.s_pose_ious.pop(frame-1)
            instance.g_pose_ious.pop(frame-1)
        except KeyError:
                pass
            
        return self.pop(instance, frame-1, starting_frame)
    
    
    def delete_old_instances(self, starting_frame):
        frame = starting_frame - 1
        idx = []
        for id, instance in self.current_instances.items():
            try:
                if instance.g_pose_ious[frame]:
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
    center = [bbox.x, bbox.y, bbox.z]
    lwh = [bbox.length, bbox.width, bbox.height]
    box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d