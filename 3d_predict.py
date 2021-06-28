import numpy as np
import math
import os
from Kmat import intrinsic

images = ['test_mono1.png']
names = []
bboxes_paths = []
depth_paths = []
box_path = './boxes/'
depth_path = './monodepth2/assets/'

for image in images:
    names.append(image.split('.')[0])

for name in names:
    bboxes_paths.append(os.path.join(box_path,str(name+'.npy')))

for name in names:
    depth_paths.append(os.path.join(depth_path,str(name+'_disp.npy')))

for path in bboxes_paths:
    count = 0
    # print(path)
    coords_bbox = np.load(path)
    # print(coords_bbox.shape)
    depth_map = np.load(depth_paths[count])
    center_x = np.array(int((coords_bbox[:,0]+coords_bbox[:,2])/2)).reshape((coords_bbox.shape[0],1))
    center_y = np.array(int((coords_bbox[:,1]+coords_bbox[:,3])/2)).reshape((coords_bbox.shape[0],1))
    K = intrinsic(ImageSizeX=1024,ImageSizeY=320)
    K_inv = K.inverse()
    final_3d = []
    for i in range(coords_bbox.shape[0]):
        coord_2d = np.array([[center_x[i]],[center_y[i]],[1]])
        coord_3d = np.matmul(K_inv,coord_2d)*depth_map[0,0,center_y[i],center_x[i]]
        final_3d.append(coord_3d)
    final_3d = np.array(final_3d)
    np.save('./results/'+str(names[count])+'_3d.npy',final_3d)
    count+=1
    print("The final 3d locations are:", final_3d)