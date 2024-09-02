# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from bbox.metrics import iou_3d
import copy
import importlib
import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import open3d as o3d
from kiss_icp.tools.utils import translate_boxes_to_open3d_instance, InstanceAssociation
from kiss_icp.tools.utils_class import BoundingBox3D, OutputPCD
from kiss_icp.tools.annotations import filter_annotations, filter_bboxes
from scipy.spatial.transform import Rotation as R

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
GREEN =  np.array([0, 1, 0])
SPHERE_SIZE = 0.20

save_mesh_dir = "results/deep_sdf/mesh"

class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, source, keypoints, target_map, pose):
        pass

class RegistrationVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
            # Suppress
            # [Open3D WARNING] invalid color in PaintUniformColor, clipping to [0, 1]
            self.o3d.utility.set_verbosity_level(self.o3d.utility.VerbosityLevel(0))
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.source = self.o3d.geometry.PointCloud()
        self.keypoints = self.o3d.geometry.PointCloud()
        self.target = self.o3d.geometry.PointCloud()
        self.frames = []
        
        # Instance association
        self.InstanceAssociation = InstanceAssociation()
        self.frames_ID = -1
        self.extracted_pcds = {}
        self.meshs = []
        self.instances = {}
        self.instances_gt = {}

        # annotations_and_detections
        self.annotations_and_detections = {}
        self.get_track_uuid = {}

        # Initialize visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_map = True
        self.render_source = True
        self.render_keypoints = False
        self.global_view = False
        self.render_trajectory = True
        # Cache the state of the visualizer
        self.state = (
            self.render_map,
            self.render_keypoints,
            self.render_source,
        )

        # Take environment file ohm edited
        self.env = None
        self.list_track_uuid = []
        self.found_instance = {}
        self.not_found_instance = {}
        self.found_instance_annotation = {}
        self.not_found_instance_annotation = {}

    def update(self, source, keypoints, target_map, pose, bboxes, annotations):
        target = target_map.point_cloud()
        self._update_geometries(source, keypoints, target, pose, bboxes, annotations)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.source)
        # self.vis.add_geometry(self.keypoints)
        # self.vis.add_geometry(self.target)
        self._set_black_background(self.vis)
        self.vis.get_render_option().point_size = 2
        self.vis.get_render_option().line_width = 20 # THIS DOES NOT WORK
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [F] to toggle on/off the input cloud to the pipeline\n"
            "\t    [K] to toggle on/off the subsbampled frame\n"
            "\t    [M] to toggle on/off the local map\n"
            "\t    [V] to toggle ego/global viewpoint\n"
            "\t    [T] to toggle the trajectory view(only available in global view)\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["V"], self._toggle_view)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["F"], self._toggle_source)
        self._register_key_callback(["K"], self._toggle_keypoints)
        self._register_key_callback(["M"], self._toggle_map)
        self._register_key_callback(["T"], self._toggle_trajectory)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_source(self, vis):
        if self.render_keypoints:
            self.render_keypoints = False
            self.render_source = True
        else:
            self.render_source = not self.render_source
        return False

    def _toggle_keypoints(self, vis):
        if self.render_source:
            self.render_source = False
            self.render_keypoints = True
        else:
            self.render_keypoints = not self.render_keypoints

        return False

    def _toggle_map(self, vis):
        self.render_map = not self.render_map
        return False

    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        self._trajectory_handle()

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_trajectory(self, vis):
        if not self.global_view:
            return False
        self.render_trajectory = not self.render_trajectory
        self._trajectory_handle()
        return False

    def _trajectory_handle(self):
        if self.render_trajectory and self.global_view:
            for frame in self.frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)
        else:
            for frame in self.frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)

    def _update_geometries(self, source, keypoints, target, pose, bboxes, annotations):
        # Source hot frame
        if self.render_source:
            self.source.points = self.o3d.utility.Vector3dVector(source)
            self.source.paint_uniform_color(YELLOW)
            if self.global_view:
                self.source.transform(pose)
        else:
            self.source.points = self.o3d.utility.Vector3dVector()

        # Keypoints
        if self.render_keypoints:
            self.keypoints.points = self.o3d.utility.Vector3dVector(keypoints)
            self.keypoints.paint_uniform_color(YELLOW)
            if self.global_view:
                self.keypoints.transform(pose)
        else:
            self.keypoints.points = self.o3d.utility.Vector3dVector()

        # Target Map
        if self.render_map:
            target = copy.deepcopy(target)
            self.target.points = self.o3d.utility.Vector3dVector(target)
            if not self.global_view:
                self.target.transform(np.linalg.inv(pose))
        else:
            self.target.points = self.o3d.utility.Vector3dVector()

        # Update always a list with all the trajectories
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(SPHERE_SIZE)
        new_frame.paint_uniform_color(BLUE)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        self.frames.append(new_frame)
        
        # Bounding Boxes, Create a box
        self.frames_ID = self.frames_ID + 1
        
        ego_car_pose = pose

        # Filter by Area
        boundingBoxes3D = []

        if self.env != None:
            x_min_bbox = self.env['bbox_size']['x_min_bbox']
            x_max_bbox = self.env['bbox_size']['x_max_bbox']
            y_min_bbox = self.env['bbox_size']['y_min_bbox']
            y_max_bbox = self.env['bbox_size']['y_max_bbox']
            boundingBoxes3D = filter_bboxes(bboxes, x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox)
            annotation_BBox3D = filter_annotations(annotations, x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox)
        else:
            boundingBoxes3D = filter_bboxes(bboxes)
            annotation_BBox3D = filter_annotations(annotations)

        # Filter by track_uuid, if no 
        if len(self.list_track_uuid) > 0:
            # Refactor to function
            new_annotations_BBox3D = []
            for i in range(len(self.list_track_uuid)):
                for annotation in annotation_BBox3D:
                    if annotation.track_uuid == self.list_track_uuid[i]:
                        new_annotations_BBox3D.append(annotation)
                        break
            annotation_BBox3D = new_annotations_BBox3D

        # Get current instance
        self.InstanceAssociation.update(ego_car_pose, boundingBoxes3D, self.frames_ID)
        current_instances = self.InstanceAssociation.get_current_instances(self.frames_ID, 3)

        # Filter instances by IOU between annotations and instances
        if True:
            new_current_instances = {}
            self.annotations_and_detections[self.frames_ID] = {}
            for id, current_instance in current_instances.items():
                if current_instance.last_frame == self.frames_ID:
                    color_code = current_instance.color_code
                    if self.global_view:
                        detected_bbox = current_instance.g_pose_ious[current_instance.last_frame]
                    else:
                        detected_bbox = current_instance.s_pose_ious[current_instance.last_frame]

                    for annotation in annotation_BBox3D:
                        annotation_bbox = BoundingBox3D(annotation.dst_SE3_object.translation[0], annotation.dst_SE3_object.translation[1],
                                                        annotation.dst_SE3_object.translation[2], annotation.length_m, 
                                                        annotation.width_m, annotation.height_m,
                                                        annotation.dst_SE3_object.rotation)
                        if iou_3d(annotation_bbox.iou, detected_bbox) > 0.3:
                            new_current_instances[id] = current_instance
                            # # when evaluate please use copy data directory to your computer
                            instance_copy = copy.deepcopy(current_instance)
                            self.annotations_and_detections[self.frames_ID][annotation.track_uuid] = tuple((instance_copy, annotation))
                            if current_instance.id not in self.get_track_uuid:
                                self.get_track_uuid[current_instance.id] = annotation.track_uuid
                            break
            
            print("self.annotations_and_detections", len(self.annotations_and_detections[self.frames_ID]))
            current_instances = new_current_instances

        # Show annotation
        for annotation in annotation_BBox3D:

            track_uuid = '1d33dc72-e987-4140-9a7f-b653b2d3b41d'
            if track_uuid not in self.extracted_pcds:
                self.extracted_pcds[track_uuid] = {}

            if 1 not in self.extracted_pcds[track_uuid]:
                self.extracted_pcds[track_uuid][1] = {}
                
            # if self.frames_ID not in self.extracted_pcds[track_uuid][self.frames_ID]:
            self.extracted_pcds[track_uuid][1][self.frames_ID] = {}

            bbox = BoundingBox3D(annotation.dst_SE3_object.translation[0], annotation.dst_SE3_object.translation[1],
                    annotation.dst_SE3_object.translation[2], annotation.length_m, 
                    annotation.width_m, annotation.height_m,
                    annotation.dst_SE3_object.rotation)

            line_set, box3d = translate_boxes_to_open3d_instance(bbox, crop=True)
            line_set.paint_uniform_color((1, 0, 0))

            if annotation.track_uuid not in self.instances_gt:
                # Create matrix
                mtx_t = np.array([bbox.x , bbox.y, bbox.z]).T
                mtx_T33 = np.hstack((bbox.rot, mtx_t[..., np.newaxis]))
                mtx_T = np.vstack((mtx_T33, np.array([0, 0, 0, 1])))

                axis_car = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
                # Update Attribute
                axis_car.transform(mtx_T)
                self.vis.add_geometry(axis_car, reset_bounding_box=False)
                
                self.vis.add_geometry(line_set, reset_bounding_box=False)

                # Store
                self.instances_gt[annotation.track_uuid] = {}
                self.instances_gt[annotation.track_uuid]['line_set'] = line_set
                self.instances_gt[annotation.track_uuid]['mtx_T'] = mtx_T
                self.instances_gt[annotation.track_uuid]['last_frame'] = self.frames_ID
                self.instances_gt[annotation.track_uuid]['axis_car'] = axis_car

                if annotation.track_uuid in self.not_found_instance_annotation:
                    del self.not_found_instance_annotation[annotation.track_uuid]
                self.found_instance_annotation[annotation.track_uuid] = annotation.track_uuid

            else:
                # Create matrix
                mtx_t = np.array([bbox.x , bbox.y, bbox.z]).T
                mtx_T33 = np.hstack((bbox.rot, mtx_t[..., np.newaxis]))
                mtx_T = np.vstack((mtx_T33, np.array([0, 0, 0, 1])))

                line_set = self.instances_gt[annotation.track_uuid]['line_set']
                prev_mtx_T = self.instances_gt[annotation.track_uuid]['mtx_T']
                mtx_T_sensor = mtx_T @ np.linalg.inv(prev_mtx_T)
                
                line_set.transform(mtx_T_sensor)
                self.vis.update_geometry(line_set)
                
                axis_car = self.instances_gt[annotation.track_uuid]['axis_car']
                axis_car.transform(mtx_T_sensor)
                self.vis.update_geometry(axis_car)

                # Store
                self.instances_gt[annotation.track_uuid]['line_set'] = line_set
                self.instances_gt[annotation.track_uuid]['mtx_T'] = mtx_T
                self.instances_gt[annotation.track_uuid]['last_frame'] = self.frames_ID
                self.instances_gt[annotation.track_uuid]['axis_car'] = axis_car
                
                if annotation.track_uuid in self.not_found_instance_annotation:
                    del self.not_found_instance_annotation[annotation.track_uuid]
                self.found_instance_annotation[annotation.track_uuid] = annotation.track_uuid


            # Extract points
            source_points_sensor = np.asarray(self.source.points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_points_sensor)
            idx_points = box3d.get_point_indices_within_bounding_box(pcd.points)
            pcd_in_bbox = np.asarray(source_points_sensor)[idx_points, :]
            self.extracted_pcds[annotation.track_uuid][1][self.frames_ID]['T_cam_obj'] = self.instances_gt[annotation.track_uuid]['mtx_T']
            print("self.", self.extracted_pcds[annotation.track_uuid][1][self.frames_ID]['T_cam_obj'])
            pcd_in_bbox_eu = pcd_in_bbox[:, :3]
            self.extracted_pcds[annotation.track_uuid][1][self.frames_ID]['pts_cam'] = pcd_in_bbox_eu
            pts_cam_homo = np.hstack((pcd_in_bbox, np.ones((pcd_in_bbox.shape[0], 1))))
            pts_obj = (np.linalg.inv(mtx_T) @ pts_cam_homo.T).T
            pts_obj = pts_obj[:, :3]
            self.extracted_pcds[annotation.track_uuid][1][self.frames_ID]['surface_points'] = pts_obj

            r = R.from_matrix(bbox.rot)
            bbox_euler = r.as_euler('zxy')[0]
            self.extracted_pcds[annotation.track_uuid][1][self.frames_ID]['bbox'] = [bbox.x, bbox.y, bbox.z,
                                                                    bbox.width, bbox.length, bbox.height, bbox_euler]
            print("self", len(self.extracted_pcds[annotation.track_uuid][1]))

        # Show current instance
        # for id, current_instance in current_instances.items():

        #     # track_uuid = self.get_track_uuid[current_instance.id]
        #     # if track_uuid not in self.extracted_pcds:
        #     #     self.extracted_pcds[track_uuid] = {}

        #     # if current_instance.id not in self.extracted_pcds[track_uuid]:
        #     #     self.extracted_pcds[track_uuid][current_instance.id] = {}
                
        #     # # if self.frames_ID not in self.extracted_pcds[track_uuid][self.frames_ID]:
        #     # self.extracted_pcds[track_uuid][current_instance.id][self.frames_ID] = {}
        #     # print("track_uuid", track_uuid, current_instance.id)

        #     if current_instance.last_frame == self.frames_ID:
        #         color_code = GREEN
        #         if self.global_view:
        #             bbox = current_instance.g_pose_visuals[current_instance.last_frame]
        #         else:
        #             bbox = current_instance.s_pose_visuals[current_instance.last_frame]
        #         line_set, box3d = translate_boxes_to_open3d_instance(bbox, crop=True)
        #         line_set.paint_uniform_color(color_code)

        #         if current_instance.id not in self.instances:
        #             # Create matrix
        #             mtx_t = np.array([bbox.x , bbox.y, bbox.z]).T
        #             mtx_T33 = np.hstack((bbox.rot, mtx_t[..., np.newaxis]))
        #             mtx_T = np.vstack((mtx_T33, np.array([0, 0, 0, 1])))

        #             # Create Attribute
        #             axis_car = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

        #             # Update Attribute
        #             axis_car.transform(mtx_T)
        #             # self.vis.add_geometry(axis_car, reset_bounding_box=False)
        #             # self.vis.add_geometry(line_set, reset_bounding_box=False)

        #             # Store
        #             self.instances[current_instance.id] = {}
        #             self.instances[current_instance.id]['axis_car'] = axis_car
        #             self.instances[current_instance.id]['line_set'] = line_set
        #             self.instances[current_instance.id]['mtx_T'] = mtx_T
        #             self.instances[current_instance.id]['last_frame'] = self.frames_ID

        #             if current_instance.id in self.not_found_instance:
        #                 del self.not_found_instance[current_instance.id]
        #             self.found_instance[current_instance.id] = current_instance.id

        #         else:
        #             # Create matrix
        #             mtx_t = np.array([bbox.x , bbox.y, bbox.z]).T
        #             mtx_T33 = np.hstack((bbox.rot, mtx_t[..., np.newaxis]))
        #             mtx_T = np.vstack((mtx_T33, np.array([0, 0, 0, 1])))

        #             # Get Attribute
        #             axis_car = self.instances[current_instance.id]['axis_car']
        #             line_set = self.instances[current_instance.id]['line_set']
        #             prev_mtx_T = self.instances[current_instance.id]['mtx_T']
        #             mtx_T_sensor = mtx_T @ np.linalg.inv(prev_mtx_T)
                    
        #             # Update Attribute
        #             axis_car.transform(mtx_T_sensor)
        #             self.vis.update_geometry(axis_car)
        #             line_set.transform(mtx_T_sensor)
        #             self.vis.update_geometry(line_set)

        #             # Store
        #             self.instances[current_instance.id]['axis_car'] = axis_car
        #             self.instances[current_instance.id]['line_set'] = line_set
        #             self.instances[current_instance.id]['mtx_T'] = mtx_T
        #             self.instances[current_instance.id]['last_frame'] = self.frames_ID
                    
        #             if current_instance.id in self.not_found_instance:
        #                 del self.not_found_instance[current_instance.id]
        #             self.found_instance[current_instance.id] = current_instance.id

        #         # # Extract points
        #         # source_points_sensor = np.asarray(self.source.points)
        #         # pcd = o3d.geometry.PointCloud()
        #         # pcd.points = o3d.utility.Vector3dVector(source_points_sensor)
        #         # idx_points = box3d.get_point_indices_within_bounding_box(pcd.points)
        #         # pcd_in_bbox = np.asarray(source_points_sensor)[idx_points, :]
        #         # self.extracted_pcds[track_uuid][current_instance.id][self.frames_ID]['T_cam_obj'] = mtx_T
        #         # pcd_in_bbox_eu = pcd_in_bbox[:, :3]
        #         # self.extracted_pcds[track_uuid][current_instance.id][self.frames_ID]['pts_cam'] = pcd_in_bbox_eu
        #         # pts_cam_homo = np.hstack((pcd_in_bbox, np.ones((pcd_in_bbox.shape[0], 1))))
        #         # pts_obj = (np.linalg.inv(mtx_T) @ pts_cam_homo.T).T
        #         # pts_obj = pts_obj[:, :3]
        #         # self.extracted_pcds[track_uuid][current_instance.id][self.frames_ID]['surface_points'] = pts_obj

        #         # r = R.from_matrix(bbox.rot)
        #         # bbox_euler = r.as_euler('zxy')[0]
        #         # self.extracted_pcds[track_uuid][current_instance.id][self.frames_ID]['bbox'] = [bbox.x, bbox.y, bbox.z,
        #         #                                                         bbox.width, bbox.length, bbox.height, bbox_euler]

        #         # #################### ADD MESH #####################
        #         # # if current_instance.id in instance_id_list:
        #         # instance_id = current_instance.id
        #         # try:
        #         #     mesh = o3d.io.read_triangle_mesh(os.path.join(f'{save_mesh_dir}velo_pts
        #         #         g_pose_path = np.load(f"results/deep_sdf/pose/new/g_pose_{instance_id}.npy", allow_pickle='TRUE').item()
        #         #         # g_pose_path = np.load(f"results/deep_sdf/pose/g_pose_{instance_id}_accumulated.npy", allow_pickle='TRUE').item()
        #         #         op_pose = g_pose_path[self.frames_ID]
        #         #     else:
        #         #         s_pose_path = np.load(f"results/deep_sdf/pose/new/s_pose_{instance_id}.npy", allow_pickle='TRUE').item()
        #         #         # s_pose_path = np.load(f"results/deep_sdf/pose/s_pose_{instance_id}_accumulated.npy", allow_pickle='TRUE').item()
        #         #         op_pose = s_pose_path[self.frames_ID]
        #         #     mesh.transform(op_pose)
        #         #     mesh.paint_uniform_color(color_code)
        #         #     self.vis.add_geometry(mesh, reset_bounding_box=False)
        #         #     self.meshs.append(mesh)
        #         # except:
        #         #     pass
        #         # #################### ADD MESH #####################

        for current_instance_id, _ in self.not_found_instance.items():
            self.vis.remove_geometry(self.instances[current_instance_id]['axis_car'], reset_bounding_box=False)
            self.vis.remove_geometry(self.instances[current_instance_id]['line_set'], reset_bounding_box=False)
            del self.instances[current_instance_id]
        self.not_found_instance = copy.deepcopy(self.found_instance)
        self.found_instance = {}

        for current_instance_id, _ in self.not_found_instance_annotation.items():
            self.vis.remove_geometry(self.instances_gt[current_instance_id]['line_set'], reset_bounding_box=False)
            del self.instances_gt[current_instance_id]
        self.not_found_instance_annotation = copy.deepcopy(self.found_instance_annotation)
        self.found_instance_annotation = {}

        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.vis.add_geometry(self.frames[-1], reset_bounding_box=False)

        self.vis.update_geometry(self.keypoints)
        self.vis.update_geometry(self.source)
        self.vis.update_geometry(self.target)
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

        # # Adjust a view -> ohm editor zoom keep changing view 
        # if self.env != None:
        #     self.vis.get_view_control().set_zoom(self.env['vis']['get_view_control']['set_zoom'])


