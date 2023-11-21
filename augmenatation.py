import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from glob import glob

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

images_names = glob('image_data/crut/*.jpg')
labels_names = glob('image_data/crut_label/*.txt')
images_names.sort()
labels_names.sort()


for i in range(len(images_names)):
    image = plt.imread(images_names[i])
    label = list(open(labels_names[i]))
    
    #이미지 내에 객체가 1개일 경우
    if len(label) == 1:
        label_list = label[0].split()
        bbox = [float(x) for x in label_list]
        class_, x, y, width, height = bbox
        bbox = [[x, y, width, height,class_]]


        transformed = transform(image=image, bboxes=bbox)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        

        transformed_bbox = setting_yolo_order(transformed_bboxes[0]) #이중리스트로 구성되어있어서 0번째 리스트를 출력하는 것임.
        transformed_bbox = np.array(transformed_bbox).reshape(1, 5)
        
        #변환 이미지 데이터, box 좌표 저장
        np.savetxt('image_data/crut_aug_label/'+'aug_cr.'+labels_names[i].split('/')[-1], transformed_bbox, delimiter = ' ', fmt = '%lf')
        array_img = Image.fromarray(transformed_image)
        array_img.save('image_data/crut_aug/'+'aug_cr.'+images_names[i].split('/')[-1])
     
        #변환 이미지 확인
        trans_xml_bboxs = yolo_to_xml_bbox(transformed_bboxes, transformed_image)
        # 이미지 출력
        plt.imshow(transformed_image)

        # bounding box 그리기
        for trans_xml_bbox in trans_xml_bboxs:
            print(trans_xml_bbox)
            xmin, ymin, xmax, ymax, category = trans_xml_bbox
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
            ax = plt.gca()
            ax.add_patch(rect)

        plt.xticks([]); plt.yticks([])
        plt.show()
        
    #이미지 내에 객채가 2개 이상일 경우
    else:
        bbox = []
        for x in range(len(label)):
            str_label = label[x].split()
            bbox_float = [float(y) for y in str_label]
            class_, x, y, width, height = bbox_float
            bbox.append([x,y,width,height,class_])
        

        transformed = transform(image=image, bboxes=bbox)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        transformed_bboxes_list = []
        for b in range(len(transformed_bboxes)):
            transformed_bboxes_list.append(setting_yolo_order(transformed_bboxes[b]))

        transformed_bbox = np.array(transformed_bboxes_list).reshape(len(transformed_bboxes_list), 5)

        np.savetxt('image_data/crut_aug_label/'+'aug_cr.'+labels_names[i].split('/')[-1], transformed_bbox, delimiter = ' ', fmt = '%lf') 
        array_img = Image.fromarray(transformed_image)
        array_img.save('image_data/crut_aug/'+'aug_cr.'+images_names[i].split('/')[-1])
        
        #변환 이미지 확인
        #객체가 많기 때문에 리스트에 각 boxing 좌표를 넣어 그리도록 한다.
        trans_xml_bboxs = []
        for xml in range(len(label)):
            trans_xml_bboxs.append(yolo_to_xml_bbox([transformed_bboxes[xml]], transformed_image))

        # 이미지 출력
        plt.imshow(transformed_image)

        # bounindg box 그리기
        for trans_xml_bbox in trans_xml_bboxs:
            print(trans_xml_bbox[0])
            
            xmin, ymin, xmax, ymax, category = trans_xml_bbox[0]
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
            ax = plt.gca()
            ax.add_patch(rect)

        plt.xticks([]); plt.yticks([])
        plt.show()