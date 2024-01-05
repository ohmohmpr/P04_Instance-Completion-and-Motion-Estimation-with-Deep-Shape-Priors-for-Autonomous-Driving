import open3d as o3d
import numpy as np
import copy

# 2D and 3D detection and data association
def main():
    points = np.load("OpenPCDet/nuscenes_point/0061_sweep/points20.npy")
    # print("points", points)
    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Create points
    pcd_transformed = o3d.geometry.PointCloud()

    # Coordinate frame
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    vis.add_geometry(pcd)
    
    vis.run()
    # vis.destroy_window()

if __name__ == "__main__":
    main()