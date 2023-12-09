import numpy as np

test = np.load("results/OpenPCDet_PointRCNN/NuScences/0061.npy", allow_pickle='TRUE').item()
# test = np.load("results/OpenPCDet_PointRCNN/KITTI/00_01.npy", allow_pickle='TRUE').item()
print(test[0])