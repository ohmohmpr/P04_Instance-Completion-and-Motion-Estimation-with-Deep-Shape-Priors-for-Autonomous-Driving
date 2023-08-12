# Contents
The goal is to make them online.
 - [1. Instance Association](#instance-association) 
 - [2. Extract Point cloud](#extract-point-cloud) 
 - [3. Optimization](#optimization) 
 - [4. Add mesh](#add-mesh) 

## Instance Association
The goal is to find data association of instances. track car.    
(in progress) current method.


### 1.1 Basic method: find iou of bounding boxes between frames.
used PointRCNN to get bounding boxed of each in car in each frame.     
then, find iou between of each car.       
if they are greater than 0. they are the same instance.      

data in object should be able to be saved.


```python
"""
could be separate box of Point3d and box of instance
"""
get_IA(bbox_of_current_frame, bbox_of_instance)

return class(Instance Association)
```

```python
class instance_association
    dict of instance
```
```python
class instance
    x, y, z, rot
```
'IndexError: index 4000 is out of bounds for axis 0 with size 4000'    
observed many of wrong instance association in sequence 00        

we can test new program on that car in sequence 04        

observed many of wrong instance association in sequence 07      
performance issue in sequence 20      

## Extract Point cloud
(offline)
```python
extract_point_cloud(source_point_from_kiss_icp, instance)
    bbox_used_to_get_pcl = create_box(instance)
    get_point_from_pcl(source_point_from_kiss_icp, bbox_used_to_get_pcl)
return pcl_of_each_instance

```

## Optimization
(offline: Due to CUDA)

```python
optimizer(pcl_of_each_instance, pose )

return latent_code

write_mesh(latent_code) (we can not create mesh below 0.1 sec)

return mesh
```
loss function in optimizer

## Add mesh
```python
addd(mesh)

```


## Next tasks
visualizer