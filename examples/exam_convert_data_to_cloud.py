from graspnetAPI import GraspNet
import cv2
import open3d as o3d
import os

# load all dataset in scene, and convert depth and color image to rgb cloud

camera = 'realsense'
# sceneId = 160
# annId = 3

####################################################################
graspnet_root = '/home/alvin/data/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

# save cloud in pcd format
cloud_dir = os.path.join(graspnet_root, 'cloud', camera)
if not os.path.exists(cloud_dir):
    os.makedirs(cloud_dir)
else:
    # remove old cloud
    cloud_files = os.listdir(cloud_dir)
    for cloud_file in cloud_files:
        os.remove(os.path.join(cloud_dir, cloud_file))

g = GraspNet(graspnet_root, camera = camera, split = 'train')

# load rgb and visualized for 1 sec
for sceneId in range(160, 190):
    for annId in range(1, 256, 8):
        bgr = g.loadBGR(sceneId = sceneId, annId = annId, camera = camera)
        cv2.imshow('bgr', bgr)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        cloud = g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = camera, align = True, use_workspace = True)

        # visualize the cloud using open3d
        # o3d.visualization.draw_geometries([cloud])

        cloud_file = os.path.join(cloud_dir, 'scene{}_{}.pcd'.format(sceneId, annId))
        o3d.io.write_point_cloud(cloud_file, cloud)