import os
import numpy as np

id = "1077"
lst = os.listdir(f"results/OpenPCDet_PointRCNN/NuScences/manual/{id}") # your directory path
number_files = len(lst) * 10
det = {}
for i in range(number_files):
    idx = int(i / 10) + 202
    points = np.load(f"results/OpenPCDet_PointRCNN/NuScences/manual/{id}/points{idx}.npy", allow_pickle='TRUE').item()

    det[i] = points[0]
    
print(len(det))
save_path = f"results/OpenPCDet_PointRCNN/NuScences/{id}-new-new.npy"

np.save(save_path, np.array(det, dtype=object), allow_pickle=True)
