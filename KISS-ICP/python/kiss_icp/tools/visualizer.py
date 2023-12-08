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
import copy
import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np
import open3d as o3d
from kiss_icp.tools.utils import translate_boxes_to_open3d_instance, InstanceAssociation
from kiss_icp.tools.utils_class import BoundingBox3D, OutputPCD
from scipy.spatial.transform import Rotation as R

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
SPHERE_SIZE = 0.20

save_mesh_dir = "results/deep_sdf/mesh"

# 1048 1049 1059
# 1066(1)
# 1075 1078(1) 1080(3w) 1081(3)
# 1090(2w) 1099(1) 1104(1)
# 1119 1120(1) 1124(1) 1129(2)
# 1131 1137(2) 1139(2) 1141(3)
instance_id_list = [0, 1]


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
        self.visual_instances = []
        self.output_pcd_s = {}
        self.meshs = []

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

    def update(self, source, keypoints, target_map, pose, bboxes, sd_token, annotation_metadata_tokens, first_pose, gt_pose):
        target = target_map.point_cloud()
        self._update_geometries(source, keypoints, target, pose, bboxes, sd_token, annotation_metadata_tokens, first_pose, gt_pose)
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
        self.vis.add_geometry(self.keypoints)
        self.vis.add_geometry(self.target)
        self._set_black_background(self.vis)
        self.vis.get_render_option().point_size = 1
        for i in range(4000):
            self.output_pcd_s[i] = {}
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

    def _update_geometries(self, source, keypoints, target, pose, bboxes, sd_token, annotation_metadata_tokens, first_pose, gt_pose):
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
        boundingBoxes3D = []

        for bbox in bboxes:
            axis_angles = np.array([0, 0, bbox[6] + 1e-10])
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            boundingBoxes3D += [BoundingBox3D(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], rot)]

        self.InstanceAssociation.update(ego_car_pose, boundingBoxes3D, self.frames_ID)
        current_instances = self.InstanceAssociation.get_current_instances(self.frames_ID, 3)

        try:
            if annotation_metadata_tokens[sd_token["sample_token"]] and sd_token["is_key_frame"] == True:
                # center_x, center_y, center_z
                translation = np.array(annotation_metadata_tokens[sd_token["sample_token"]]["translation"])
                translation = translation[:, np.newaxis]
                # width, length, height.
                size = annotation_metadata_tokens[sd_token["sample_token"]]["size"]
                lhw_size = [size[1], size[0], size[2]]

                # quaternion: w, x, y, z.
                rotation = annotation_metadata_tokens[sd_token["sample_token"]]["rotation"]
                r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])

                global_pose = np.hstack((r.as_matrix(), translation))
                global_pose = np.vstack((global_pose, np.array([0, 0, 0, 1])))

                poses = np.linalg.inv(gt_pose) @ np.linalg.inv(first_pose) @ global_pose
                bbox = BoundingBox3D(poses[0, 3], poses[1, 3], poses[2, 3], 
                                    lhw_size[0], lhw_size[1], lhw_size[2], poses[:3, :3])

                # x, y, z, l, w, h, rotation_matrix
                line_set, box3d = translate_boxes_to_open3d_instance(bbox)
                line_set.paint_uniform_color((1, 0, 0))
                self.vis.add_geometry(line_set, reset_bounding_box=False)

        except KeyError:
            pass
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)

        # Visual in sensor frame
        self.remove_all()

        
        
        for id, current_instance in current_instances.items():
            # if current_instance.last_frame == self.frames_ID and current_instance.id in instance_id_list:
            
            
            if current_instance.last_frame == self.frames_ID:
                ##################### VISUALIZAION ALL #####################
                color_code = current_instance.color_code
                
                if self.global_view:
                    bbox = current_instance.g_pose_visuals[current_instance.last_frame]
                else:
                    bbox = current_instance.s_pose_visuals[current_instance.last_frame]
                    
                line_set, box3d = translate_boxes_to_open3d_instance(bbox)
                line_set.paint_uniform_color(color_code)
                self.vis.add_geometry(line_set, reset_bounding_box=False)
                self.visual_instances.append(line_set)
                ##################### VISUALIZAION ALL #####################
                
                
            #     ##################### VISUALIZAION SOME #####################
            # # if current_instance.last_frame == self.frames_ID:
            # if current_instance.last_frame == self.frames_ID and (self.frames_ID >= 900 and self.frames_ID <= 950) \
            #     or (current_instance.id >= 1048 and current_instance.id <= 1147):
            #     instance_id = current_instance.id
            #     color_code = current_instance.color_code
                
            #     if self.global_view:
            #         bbox = current_instance.g_pose_visuals[current_instance.last_frame]
            #     else:
            #         bbox = current_instance.s_pose_visuals[current_instance.last_frame]
                    
            #     line_set, box3d = translate_boxes_to_open3d_instance(bbox)
            #     line_set.paint_uniform_color(color_code)
            #     self.vis.add_geometry(line_set, reset_bounding_box=False)
            #     self.visual_instances.append(line_set)
                
            #     #################### VISUALIZAION SOME #####################
                
                # #################### EXTRACT POINTS #####################
                # # Get s_pose_visuals
                # g_selected_bbox = current_instance.g_pose_visuals[current_instance.last_frame]
                # s_selected_bbox = current_instance.s_pose_visuals[current_instance.last_frame]
                # _, box3d = translate_boxes_to_open3d_instance(s_selected_bbox, True)
                
                # # Get s_source_points
                # s_source_points = np.asarray(self.source.points)
                # # source_points = np.hstack((np.asarray(self.source.points), np.ones((n.p.asarray(self.source.points).shape[0], 1))))
                # # g_source_points = (pose @ source_points.T).T
                # # g_source_points = g_source_points[:, :3]
                
                # # Create points
                # pcd = o3d.geometry.PointCloud()
                # # From numpy to Open3D
                # pcd.points = o3d.utility.Vector3dVector(s_source_points)

                # # Get index 
                # idx_points = box3d.get_point_indices_within_bounding_box(pcd.points)
                
                # # PCL in Bbox
                # point_bbox = np.asarray(s_source_points)[idx_points, :]
                # self.output_pcd_s[instance_id][self.frames_ID] = OutputPCD(point_bbox, g_selected_bbox, s_selected_bbox)
                # #################### EXTRACT POINTS #####################
                
                # ##################### EXTRACT POINTS #####################
                # filename = f'results/instance_association/new/PointCloud_KITTI21_Obj_ID_{instance_id}-test.npy'
                # with open(filename, 'wb') as f:
                #     np.save(f, np.array(self.output_pcd_s[instance_id], dtype=object), allow_pickle=True)
                # ##################### EXTRACT POINTS #####################
                
                # #################### ADD MESH #####################
                # # if current_instance.id in instance_id_list:
                # instance_id = current_instance.id
                # try:
                #     mesh = o3d.io.read_triangle_mesh(os.path.join(f'{save_mesh_dir}/new/{instance_id}', "%d.ply" % self.frames_ID))
                #     # mesh = o3d.io.read_triangle_mesh(os.path.join(f'{save_mesh_dir}/{instance_id}-accumulated', "%d.ply" % self.frames_ID))
                #     mesh.compute_vertex_normals()
                    
                #     if self.global_view:
                #         g_pose_path = np.load(f"results/deep_sdf/pose/new/g_pose_{instance_id}.npy", allow_pickle='TRUE').item()
                #         # g_pose_path = np.load(f"results/deep_sdf/pose/g_pose_{instance_id}_accumulated.npy", allow_pickle='TRUE').item()
                #         op_pose = g_pose_path[self.frames_ID]
                #     else:
                #         s_pose_path = np.load(f"results/deep_sdf/pose/new/s_pose_{instance_id}.npy", allow_pickle='TRUE').item()
                #         # s_pose_path = np.load(f"results/deep_sdf/pose/s_pose_{instance_id}_accumulated.npy", allow_pickle='TRUE').item()
                #         op_pose = s_pose_path[self.frames_ID]
                #     mesh.transform(op_pose)
                #     mesh.paint_uniform_color(color_code)
                #     self.vis.add_geometry(mesh, reset_bounding_box=False)
                #     self.meshs.append(mesh)
                # except:
                #     pass
                # #################### ADD MESH #####################
                
                
        
        
        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.vis.add_geometry(self.frames[-1], reset_bounding_box=False)

        self.vis.update_geometry(self.keypoints)
        self.vis.update_geometry(self.source)
        self.vis.update_geometry(self.target)
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

    def remove_all(self):
        length = len(self.visual_instances)
        length_mesh = len(self.meshs)
        
        for i in range(length):
            self.vis.remove_geometry(self.visual_instances[0], reset_bounding_box=False)
            self.visual_instances.pop(0)
        
        for i in range(length_mesh):
            self.vis.remove_geometry(self.meshs[0], reset_bounding_box=False)
            self.meshs.pop(0)