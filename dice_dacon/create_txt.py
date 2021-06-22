import os
import cv2
import mmcv
from os import listdir
import json
import numpy as np

EO_CLASSES = ['container', 'oil_tanker', 'aircraft_carrier', 'maritime_vessels']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def load_image(image_id):
    img = cv2.imread(image_id)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def load_annotations(json_data, image_id):
    annotations = np.zeros((0, 10))
    
    for e in json_data['features'] :
        if (e['properties']["image_id"] == image_id):
            annotation = np.zeros((1, 10))
            Bbox_Coords = e['properties']["bounds_imcoords"].split(',')
            annotation[:, :8] = Bbox_Coords
            annotation[:, 8] = e['properties']["type_id"] - 1
            annotation[:, 9] = 0
            
            annotations = np.append(annotations, annotation, axis=0)
    
    return annotations

image_dir = "/data/hdd/Dacon/dota/train/images"
json_dir = "/data/hdd/Dacon/dota/train/labels.json"
image_filenames = [os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
gsd = "gsd:0.11\n"
imagesource = "imagesource:GoogleEarth\n"
txt_file = "/data/hdd/Dacon/dota/train/new_labelTxt"
new_image_dir = "/data/hdd/Dacon/dota/train/new_images"

ratio = [1]

with open(json_dir) as json_file:
    json_data = json.load(json_file)

print("# image :", len(image_filenames))

for index in range(len(image_filenames)):
    image_id = image_filenames[index]
    img = load_image(image_id)
    for t in range(len(ratio)):
        ratio_image = cv2.resize(img, dsize=(0, 0), fx=ratio[t], fy=ratio[t], interpolation=cv2.INTER_CUBIC)
        image_name = image_id.split('/')[7]
        anno = load_annotations(json_data, image_name)
        txt_name = image_name.split('.')[0] + '_' + str(ratio[t]) + '.txt'
        new_name = image_name.split('.')[0] + '_' + str(ratio[t]) + '.png'

        if(len(anno) == 0):
            print(image_name)
            os.remove(image_id) # 오브젝이 없는 이미지는 제거
            t = 100
            continue
        
        cv2.imwrite(os.path.join(new_image_dir, new_name), ratio_image)

        f = open(os.path.join(txt_file, txt_name), 'w')
        f.write(imagesource)
        f.write(gsd)

        for i in range(len(anno)):
            for j in range(10):
                if(j == 8): # class
                    space = EO_CLASSES[int(anno[i][j])] + ' '
                elif(j == 9): # difficult
                    space = str(int(anno[i][j])) + ' '
                else:
                    space = str(anno[i][j] * ratio[t]) + ' '
                f.write(space)
            f.write('\n')

    if(index % 50 == 0):
        print(index)
        
    