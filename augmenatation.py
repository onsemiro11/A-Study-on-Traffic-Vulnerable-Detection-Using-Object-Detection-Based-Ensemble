import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from tqdm import   tqdm
from glob import glob
from PIL import Image

#yolo 모델에 맞는 좌표로 되어 있는 좌표 데이터들을 잠시 augmentation하기 위해 x,y좌표로 변환해주는 함수
def yolo_to_xml_bbox(bbox, image): 
    # x_center, y_center width, heigth
    h, w, c = image.shape
    w_half_len = (bbox[0][2] * w) / 2
    h_half_len = (bbox[0][3] * h) / 2
    xmin = int((bbox[0][0] * w) - w_half_len)
    ymin = int((bbox[0][1] * h) - h_half_len)
    xmax = int((bbox[0][0] * w) + w_half_len)
    ymax = int((bbox[0][1] * h) + h_half_len)
    im_class = int(int(bbox[0][-1]))
    return [(xmin, ymin, xmax, ymax, im_class)]


def setting_yolo_order(list_):
    a, b, c, d,e = list_
    return [e,a,b,c,d]

transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1), #밝기 정도
            A.Blur(p=1, blur_limit=(5, 10)), #흐림 정도
            A.Affine(
                         translate_px={'x':(-10, 10), 'y':(-10, 10)},
                         scale = (0.5, 1),
                         rotate=(10, 10)) #affine 변환 - 위치 회전 (선형변환에 위치까지 변환시킨다)
        ], bbox_params=A.BboxParams(format = 'yolo'))

images_names = glob('data/images/*.png')
labels_names = glob('data/labels/*.txt')
images_names.sort()
labels_names.sort()

print(f"Before image file : {len(images_names)}, label file : {len(labels_names)}")

for i in tqdm(range(999,len(images_names))):

    # image format : jpg
    # image = plt.imread(images_names[i])

    # image format : png
    image = Image.open(images_names[i])
    image = np.array(image)

    label = list(open(labels_names[i]))

    bbox = []
    for x in range(len(label)):
        str_label = label[x].split()
        bbox_float = [float(y) for y in str_label]
        class_, x, y, width, height = bbox_float
        bbox.append([x,y,width,height,class_])
    
    try:
        transformed = transform(image=image, bboxes=bbox)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_bboxes_list = []
        for b in range(len(transformed_bboxes)):
            transformed_bboxes_list.append(setting_yolo_order(transformed_bboxes[b]))

        transformed_bbox = np.array(transformed_bboxes_list).reshape(len(transformed_bboxes_list), 5)

        np.savetxt('data/labels/'+'aug_'+labels_names[i].split('/')[-1], transformed_bbox, delimiter = ' ', fmt = '%lf') 
        array_img = Image.fromarray(transformed_image)
        array_img.save('data/images/'+'aug_'+images_names[i].split('/')[-1])


    except ValueError:
        pass

images_names = glob('data/images/*.png')
labels_names = glob('data/labels/*.txt')

print(f"After image file : {len(images_names)}, label file : {len(labels_names)}")