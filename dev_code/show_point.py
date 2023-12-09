import open3d as o3d
import numpy as np
import copy

# 2D and 3D detection and data association
def main():
    points = np.load("data/nuscenes_point/points381.npy")
    points_transformed = np.load("data/nuscenes_point_transformed/points381.npy")
    print("points", points.shape)
    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Create points
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(points[:, :3])
    # R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
    # pcd.rotate(R, center=(0, 0, 0))

    # Coordinate frame
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    vis.add_geometry(pcd)
    vis.add_geometry(pcd_transformed)
    
    vis.run()
    # vis.destroy_window()

if __name__ == "__main__":
    main()