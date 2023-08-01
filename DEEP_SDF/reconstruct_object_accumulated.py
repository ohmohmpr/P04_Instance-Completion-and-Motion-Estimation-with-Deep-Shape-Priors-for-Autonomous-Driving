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
    
    id = 512
    detections = np.load(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy', allow_pickle='TRUE').item()

    # start reconstruction
    objects_recon = {}
    
    # Ugly initialization
    g_point = {}
    g_point_0 = np.array([[0, 0, 0]])
    
    # Mesh
    save_dir = cfg['save_mesh_dir']
    
    _, rott = convert_to_world_frame(g_point_0)
    
    start = get_time()

    g_pose = {}
    s_pose = {}
    
    is_first_f = False
    len_accu_points = 0
    for frame_id, det in detections.items():
        if det.canonical_point.shape[0] > 200:
        # if det.canonical_point.shape[0] > 200 or len_accu_points > 200:
            
            if is_first_f:
                accumulated_canonical_point = np.concatenate((prev_point_for_optim, det.canonical_point))
                c_all_points = convert_to_canonic_space(accumulated_canonical_point)
                len_accu_points = accumulated_canonical_point.shape[0]
                
                tmp_canonical_point = np.hstack((accumulated_canonical_point, np.ones((accumulated_canonical_point.shape[0], 1))))
                g_space_point = (det.g_pose @ tmp_canonical_point.T).T
                acc_g_point = np.concatenate((g_point_0, g_space_point[:, :3]))
                
            else: 
                c_all_points = convert_to_canonic_space(det.canonical_point)
                is_first_f = True
                
                tmp_canonical_point = np.hstack((det.canonical_point, np.ones((det.canonical_point.shape[0], 1))))
                g_space_point = (det.g_pose @ tmp_canonical_point.T).T
                acc_g_point = np.concatenate((g_point_0, g_space_point[:, :3]))
                
            obj = optimizer.reconstruct_object(np.eye(4) , c_all_points)
            
            prev_point_for_optim = obj.inv_g_space_point
            
            obj.g_pose = det.g_pose @ rott @ obj.t_cam_obj
            obj.s_pose = det.s_pose @ rott @ obj.t_cam_obj
            
            g_pose[frame_id] = obj.g_pose
            s_pose[frame_id] = obj.s_pose
            
            # print(g_pose)
            # print(s_pose)

            objects_recon[frame_id] = obj
            

            g_point[frame_id] = acc_g_point
    
    np.save(f'results/deep_sdf/pose/g_pose_{id}_accumulated.npy', np.array(g_pose, dtype=object), allow_pickle=True)
    np.save(f'results/deep_sdf/pose/s_pose_{id}_accumulated.npy', np.array(s_pose, dtype=object), allow_pickle=True)
    
    c_all_points = c_all_points[:, :3]
    c_all_points, rott = convert_to_world_frame(c_all_points)
    g_point_0 = g_point_0[1:]
    
    
    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add Source LiDAR point cloud
    c_source_pcd = o3d.geometry.PointCloud()
    c_source_pcd.points = o3d.utility.Vector3dVector(c_all_points)
    BLUE_color = np.full((c_all_points.shape[0], 3), color_table[2]) # BLUE COLOR
    c_source_pcd.colors = o3d.utility.Vector3dVector(BLUE_color)
    vis.add_geometry(c_source_pcd)
    
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for (frame_id, obj), (_, g_point_obj) in zip(objects_recon.items(), g_point.items()):
        
        # Add Source LiDAR point cloud - global
        # g_source_pcd = o3d.geometry.PointCloud()
        # g_source_pcd.points = o3d.utility.Vector3dVector(g_point_obj)
        # green_color = np.full((g_point_obj.shape[0], 3), color_table[1]) # GREEN COLOR
        # g_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
        # # g_source_pcd.transform(obj.g_pose)
        # vis.add_geometry(g_source_pcd)
        
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])

        # Transform mesh from object to world coordinate
        mesh_o3d.transform(obj.g_pose)
        vis.add_geometry(mesh_o3d)
        
        write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}-accumulated", "%d.ply" % frame_id))
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()

# python3 DEEP_SDF/reconstruct_object_accumulated.py

if __name__ == "__main__":
    main()