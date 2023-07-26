import numpy as np
import open3d as o3d

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