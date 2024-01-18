from bbox import  BBox3D
import numpy as np
import open3d as o3d
from typing import Callable, List

from kiss_icp.tools.utils_class import BoundingBox3D

def filter_annotations(annotations, x_min=-500, x_max=500, y_min=-500, y_max=500) -> List:
    annotations_BBox3D = []
    for annotation in annotations:
        x = annotation.dst_SE3_object.translation[0]
        y = annotation.dst_SE3_object.translation[1]
        if (x > x_min and x < x_max) and (y > y_min and y < y_max):
            annotations_BBox3D.append(annotation)

    return annotations_BBox3D

def filter_bboxes(bboxes, x_min=-500, x_max=500, y_min=-500, y_max=500) -> List:
    bboxes_BBox3D = []
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        if (x > x_min and x < x_max) and (y > y_min and y < y_max):
            axis_angles = np.array([0, 0, bbox[6] + 1e-10])
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            bboxes_BBox3D += [BoundingBox3D(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], rot)]

    return bboxes_BBox3D