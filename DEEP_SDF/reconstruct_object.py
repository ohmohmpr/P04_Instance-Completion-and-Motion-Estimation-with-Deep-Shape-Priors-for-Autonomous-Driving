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
from reconstruct.optimizer import Optimizer, MeshExtractor
from reconstruct.utils import color_table, write_mesh_to_ply
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
    g_point = np.array([[0, 0, 0]])
    
    # Mesh
    save_dir = cfg['save_mesh_dir']
    
    _, rott = convert_to_world_frame(g_point)
    
    start = get_time()

    g_pose = {}
    s_pose = {}
    
    for frame_id, det in detections.items():
        if det.canonical_point.shape[0] > 200:
            c_all_points = convert_to_canonic_space(det.canonical_point)
            obj = optimizer.reconstruct_object(np.eye(4) , c_all_points)
            obj.g_pose = det.g_pose @ rott @ obj.t_cam_obj
            obj.s_pose = det.s_pose @ rott @ obj.t_cam_obj
            
            g_pose[frame_id] = obj.g_pose
            s_pose[frame_id] = obj.s_pose
            
            print(g_pose)
            print(s_pose)

            objects_recon[frame_id] = obj
            g_point = np.concatenate((g_point, det.PCD))
    
    np.save('results/deep_sdf/pose/g_pose_512.npy', np.array(g_pose, dtype=object), allow_pickle=True)
    np.save('results/deep_sdf/pose/s_pose_512.npy', np.array(s_pose, dtype=object), allow_pickle=True)
    
    g_point = g_point[1:]
    
    
    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # # Add Source LiDAR point cloud
    # c_source_pcd = o3d.geometry.PointCloud()
    # c_source_pcd.points = o3d.utility.Vector3dVector(canonical_point)
    # green_color = np.full((canonical_point.shape[0], 3), color_table[1]) # Green COLOR
    # c_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
    # vis.add_geometry(c_source_pcd)
    
    # Add Source LiDAR point cloud - global
    g_source_pcd = o3d.geometry.PointCloud()
    g_source_pcd.points = o3d.utility.Vector3dVector(g_point)
    green_color = np.full((g_point.shape[0], 3), color_table[2]) # BLUE COLOR
    g_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
    vis.add_geometry(g_source_pcd)
    
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for frame_id, obj in objects_recon.items():
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])

        # Transform mesh from object to world coordinate
        mesh_o3d.transform(obj.s_pose)
        vis.add_geometry(mesh_o3d)
        
        write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}", "%d.ply" % frame_id))
        
        
    # must be put after adding geometries
    # set_view(vis, dist=20, theta=0.)
    
    idx = int(mesh.vertices.shape[0]/2)
    print("mesh.vertices", idx)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()




















def convert_to_canonic_space(pcd_g_space):
    '''
    canonical space and global frame have different origin frame
    '''
    x_angle = np.deg2rad(-90)
    z_angle = np.deg2rad(90)

    rot_x_world = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle), 0],
        [0, np.sin(x_angle),  np.cos(x_angle), 0],
        [0, 0, 0, 1]
    ])

    rot_z_world = np.array([
        [np.cos(z_angle), -np.sin(z_angle),0 ,0],
        [np.sin(z_angle),  np.cos(z_angle),0 ,0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    scale_pam = 1./2.1
    scale = scale_pam * np.array(np.eye(3))
        
    rott = rot_x_world @ rot_z_world

    scale = scale @ rott[:3, :3]


    rott_temp = np.hstack((scale, rott[0:3, 3][np.newaxis].T))
    rott = np.vstack((rott_temp, rott[3]))
    
    pcd_g_space = np.hstack((pcd_g_space, np.ones((pcd_g_space.shape[0], 1))))
    
    pcd_c_space = (rott @ pcd_g_space.T).T
    return pcd_c_space[:, :3]


def convert_to_world_frame(pcd_c_space):
    '''
    canonical space and global frame have different origin frame
    '''
    x_angle = np.deg2rad(-90)
    z_angle = np.deg2rad(90)

    rot_x_world = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle), 0],
        [0, np.sin(x_angle),  np.cos(x_angle), 0],
        [0, 0, 0, 1]
    ])

    rot_z_world = np.array([
        [np.cos(z_angle), -np.sin(z_angle),0 ,0],
        [np.sin(z_angle),  np.cos(z_angle),0 ,0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    scale_pam = 1./2.1
    scale = scale_pam * np.array(np.eye(3))
        
    rott = rot_x_world @ rot_z_world

    scale = scale @ rott[:3, :3]


    rott_temp = np.hstack((scale, rott[0:3, 3][np.newaxis].T))
    rott = np.linalg.inv(np.vstack((rott_temp, rott[3])))
    
    
    pcd_c_space = np.hstack((pcd_c_space, np.ones((pcd_c_space.shape[0], 1))))
    
    pcd_g_space = (rott @ pcd_c_space.T).T
    return pcd_g_space[:, :3], rott

if __name__ == "__main__":
    main()