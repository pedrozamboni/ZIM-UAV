import laspy 
import numpy as np 
import open3d as o3d
import icp_v1

### open uav roof pcd 
infile = laspy.read('/home/pedro/Documents/Documents/projects/pointcloud/Fusion_uav_als/pcd_segmented/xg09_rf.laz')
np_pcd = np.vstack((infile.x, infile.y, infile.z,infile.red,infile.green,infile.blue,infile.ensemble)).transpose()
np_roof = np_pcd[[np.where(np_pcd[:,-1]==2)[0]],0:3][0]
np_street = np_pcd[[np.where(np_pcd[:,-1]==1)[0]],0:3][0]
print('Load drone pcd')
### open target
target = o3d.io.read_point_cloud("/home/pedro/Documents/Documents/projects/pointcloud/Fusion_uav_als/Poppenhausen_example_ALS_ground_roofs.ply")
print('Load als pcd')

### open ground points

ground_file = laspy.read('/home/pedro/Documents/Documents/projects/pointcloud/Fusion_uav_als/ground.laz')
ground_pcd  = np.vstack((ground_file.x, ground_file.y, ground_file.z,ground_file.blue, ground_file.green, ground_file.blue)).transpose()

ground =  o3d.geometry.PointCloud()
ground.points = o3d.utility.Vector3dVector(ground_pcd[:,0:3])
print('Load ground pcd')


### uising only roof
## using only roof as source 
initial_tranformation = np.eye(4)
#initial_tranformation[2, 3] = -48

for p in [np_street,np_roof]:

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(p)
    max_correspondence_distance = 0.001
    trandromation = icp_v1.icp(source, target, initial_tranformation,max_correspondence_distance,'point2plane')
    initial_tranformation = trandromation.transformation

# ground.transform(trandromation.transformation)

# ### saving
# with laspy.open('/home/pedro/Documentos/Documents/projects/pointcloud/Fusion_uav_als/ground_plane2plane.laz',
#                         mode="w", header=infile.header) as writer:
        
#         point_record = laspy.ScaleAwarePointRecord.zeros(np.asarray(ground.points).shape[0], header=infile.header)
#         point_record.x = np.asarray(ground.points)[:,0]
#         point_record.y = np.asarray(ground.points)[:,1]
#         point_record.z = np.asarray(ground.points)[:,2]
        
#         point_record.red = ground_pcd[:,3]
#         point_record.green = ground_pcd[:,4]
#         point_record.blue = ground_pcd[:,5]
        
#         #point_record.ensemble = np_pcd[:,-1]

        
#         writer.write_points(point_record)