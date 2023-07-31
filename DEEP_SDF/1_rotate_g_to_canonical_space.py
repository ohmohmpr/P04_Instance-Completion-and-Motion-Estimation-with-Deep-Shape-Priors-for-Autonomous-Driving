import numpy as np

id = 1
bounding_boxes = np.load(f"results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy", allow_pickle='TRUE').item()
new_one = {}

thes_hold_num_pcd = 200

for k, car in bounding_boxes.items():
    
    if car.PCD.shape[0] > thes_hold_num_pcd:
        pose = np.hstack((car.g_pose_visuals.rot, np.array([[car.g_pose_visuals.x], [car.g_pose_visuals.y], [car.g_pose_visuals.z]])))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        pcd = np.hstack((car.PCD, np.ones((car.PCD.shape[0] , 1))))
        inv_pose = np.linalg.inv(pose)
        bounding_boxes = (inv_pose @ pcd.T).T
        bounding_boxes = bounding_boxes[:, :3]
        new_one[k] = bounding_boxes
        
        # np.savetxt(f"results/instance_association/{id}-backup/{k}.txt", bounding_boxes)
        np.save('new_one.npy', np.array(new_one, dtype=object), allow_pickle=True)
        # np.savetxt(f"results/instance_association/{id}-backup/{k}.txt", bounding_boxes)