import numpy as np
import math
import os

images = ['test_mono1.png']
names = []
bboxes_paths = []
base_path = './yolov3/runs/detect/exp2/labels'
for image in images:
    names.append(image.split('.')[0])

for name in names:
    bboxes_paths.append(os.path.join(base_path,str(name+'.txt')))

for path in bboxes_paths:
    count = 0
    print(path)
    with open(path, 'r') as f:
        bboxes = []
        for line in f:
            vals = line.split()
            temp = vals[1:]
            coords = list(map(int, temp))
            bboxes.append(coords)
    image_boxes = np.array(bboxes)
    np.save('./boxes/'+str(names[count])+'.npy',image_boxes)
    count+=1