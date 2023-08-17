#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#
import time
import click
from deep_sdf.deep_sdf.workspace import config_decoder
import open3d as o3d
import numpy as np
import os
from os.path import join, dirname, abspath
from reconstruct.loss_utils import get_time
from reconstruct.optimizer_accumulated import Optimizer, MeshExtractor
from reconstruct.utils import color_table, write_mesh_to_ply, convert_to_world_frame, convert_to_canonic_space
import yaml

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/config.yaml'))


# 2D and 3D detection and data association
def main(config):
    cfg = yaml.safe_load(open(config))
    DeepSDF_DIR = cfg['deepsdf_dir']
    decoder = config_decoder(DeepSDF_DIR)
    
    optimizer = Optimizer(decoder, cfg)
    # instance_id_list = [0, 1, 72, 209, 373, 512, 551, 555]
    id = 373
    detections = np.load(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy', allow_pickle='TRUE').item()

    # start reconstruction
    objects_recon = {}
    
    # Ugly initialization
    g_point = np.array([[0, 0, 0]])
    
    # Mesh
    save_dir = cfg['save_mesh_dir']
    is_first_f = True

    _, rott = convert_to_world_frame(g_point)
    
    start = get_time()

    g_pose = {}
    s_pose = {}
    
    hasCode = False
    iter = 0
    for frame_id, det in detections.items():
        if det.pts_obj_global.shape[0] > 200:
            iter = iter + 1
            if iter <= 0:
                pass
            else:
                ####################################### NOISE #######################################
                pts_canonical_space_homo = np.hstack((det.pts_obj_global, np.ones((det.pts_obj_global.shape[0], 1))))

                theta = np.deg2rad(30)
                print("theta", theta)
                rot_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]])
                
                pts_canonical_space_homo_op = (rot_z @ pts_canonical_space_homo.T).T
                
                det.pts_obj_global = pts_canonical_space_homo_op[:, :3]
                ####################################### NOISE #######################################

            ################# 1. Optimize using pts in bbox #################
            pts_canonical_space = convert_to_canonic_space(det.pts_obj_global)
            if hasCode == False:
                hasCode = True
                obj = optimizer.reconstruct_object(np.eye(4) , pts_canonical_space)
            else:
                obj = optimizer.reconstruct_object(np.eye(4) , pts_canonical_space, code)

            # Apply optimized transformation to point cloud in homogenous coord.
            obj.pts_canonical_space_opt = homo_trans(obj.t_cam_obj, pts_canonical_space)
            ################# 1. Optimize using pts in bbox #################

            ################# 2. Accumulated #################
            if is_first_f:
                is_first_f = False   
                pts_canonical_space_opt_accu = obj.pts_canonical_space_opt
            else:
                pts_canonical_space_opt_accu = np.concatenate((pts_canonical_space_opt_accu, obj.pts_canonical_space_opt))
                print("pts_canonical_space_opt_accu", pts_canonical_space_opt_accu.shape[0])
            ################# 2. Accumulated #################

            ################# 3. Get code from accumulated bbox #################
            obj_accu = optimizer.reconstruct_object(np.eye(4) , pts_canonical_space_opt_accu)
            code = obj_accu.code

            # Apply optimized transformation to point cloud in homogenous coord.
            obj_accu.pts_canonical_space_opt_accu_opt = homo_trans(obj.t_cam_obj, pts_canonical_space)
            ################# 3. Get code from accumulated bbox #################

            ################# 4. SEND DATA #################
            obj_accu.pts_canonical_space = pts_canonical_space
            obj_accu.t_cam_obj_opt = obj.t_cam_obj

            # World frame
            obj_accu.g_pose = det.g_pose
            obj_accu.g_pose_mesh = det.g_pose @ rott @ obj_accu.t_cam_obj_opt
            obj_accu.s_pose = det.s_pose
            obj_accu.s_pose_mesh = det.s_pose @ rott @ obj_accu.t_cam_obj_opt
            g_pose[frame_id] = obj_accu.g_pose_mesh
            s_pose[frame_id] = obj_accu.s_pose_mesh

            pts_opt_world_space, _ = convert_to_world_frame(obj_accu.pts_canonical_space_opt_accu_opt)
            obj_accu.pts_opt_world_space_opt = pts_opt_world_space

            objects_recon[frame_id] = obj_accu
            ################# 4. SEND DATA #################

    # # np.save(f'results/deep_sdf/pose/g_pose_{id}.npy', np.array(g_pose, dtype=object), allow_pickle=True)
    # # np.save(f'results/deep_sdf/pose/s_pose_{id}.npy', np.array(s_pose, dtype=object), allow_pickle=True)
    
    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for (frame_id, obj) in objects_recon.items():
        ##################### CANONICAL SPACE #####################

        #### THE OPTIMIZED ONE ####
        pts_canonical_space_opt_accu_opt = o3d.geometry.PointCloud()
        pts_canonical_space_opt_accu_opt.points = o3d.utility.Vector3dVector(obj.pts_canonical_space_opt_accu_opt)
        blue_color = np.full((obj.pts_canonical_space_opt_accu_opt.shape[0], 3), color_table[2]) # BLUE COLOR
        pts_canonical_space_opt_accu_opt.colors = o3d.utility.Vector3dVector(blue_color)
        vis.add_geometry(pts_canonical_space_opt_accu_opt)
        #### THE OPTIMIZED ONE ####

        #### THE INITIAL Guess ONE ####
        pts_canonical_space = o3d.geometry.PointCloud()
        pts_canonical_space.points = o3d.utility.Vector3dVector(obj.pts_canonical_space)
        green_color = np.full((obj.pts_canonical_space.shape[0], 3), color_table[1]) # GREEN COLOR
        pts_canonical_space.colors = o3d.utility.Vector3dVector(green_color)
        vis.add_geometry(pts_canonical_space)
        #### THE INITIAL Guess ONE ####

        #### MESH OF THE OPTIMIZED ONE ####
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])
        mesh_o3d.transform(obj.t_cam_obj_opt)
        vis.add_geometry(mesh_o3d)
        #### MESH OF THE OPTIMIZED ONE ####


        time.sleep(10)
        vis.remove_geometry(pts_canonical_space_opt_accu_opt)
        vis.remove_geometry(pts_canonical_space)
        vis.remove_geometry(mesh_o3d)
        ##################### CANONICAL SPACE #####################

        # ##################### GLOBAL SPACE #####################
        # #### THE OPTIMIZED ONE ####
        # pts_opt_world_space_opt = o3d.geometry.PointCloud()
        # pts_opt_world_space_opt.points = o3d.utility.Vector3dVector(obj.pts_opt_world_space_opt)
        # green_color = np.full((obj.pts_opt_world_space_opt.shape[0], 3), color_table[2]) # BLUE COLOR
        # pts_opt_world_space_opt.colors = o3d.utility.Vector3dVector(green_color)
        # pts_opt_world_space_opt.transform(obj.g_pose)
        # vis.add_geometry(pts_opt_world_space_opt)
        # #### THE OPTIMIZED ONE ####


        # #### MESH OF THE OPTIMIZED ONE ####
        # mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        # mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        # mesh_o3d.compute_vertex_normals()
        # mesh_o3d.paint_uniform_color(color_table[0])

        # # Transform mesh from object to world coordinate
        # mesh_o3d.transform(obj.g_pose_mesh)
        # vis.add_geometry(mesh_o3d)
        # #### MESH OF THE OPTIMIZED ONE ####
        # ##################### GLOBAL SPACE #####################

        # o3d.io.write_point_cloud(f"results/deep_sdf/pcd/{id}-pcd-canonic-accu/{frame_id}.pcd", pts_opt_world_space)
        # write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}", "%d.ply" % frame_id))
    
    print("FINISHED")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()

def homo_trans(transformation, pts):
    pts_homo = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_homo_op = (transformation @ pts_homo.T).T
    pts_op = pts_homo_op[:, :3]

    return pts_op


if __name__ == "__main__":
    main()