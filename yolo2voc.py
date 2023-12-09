import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import cv2
from sklearn.model_selection import train_test_split


class make_voc():

    def __init__(self,file_name, im_format, save_img):
        self.file_name = file_name
        self.im_format = im_format
        self.save_img = save_img
        self.object_name = {'Car':'2','Cyclist':'1','Truck':'7','Pedestrian':'0','Person_sitting':'0','Van':'4','Tram':'6'}


    def convert_voc_to_yolo(self,size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[2])/2.0 - 1
        y = (box[1] + box[3])/2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)


    def main(self):

        for file in tqdm(self.file_name):

            yolo_coor = []
            txt_file = open(file,'r')
            txt_label = txt_file.readlines()
            
            image = cv2.imread(file.replace('txt',self.im_format))
            img_shape = image.shape

            for w_d in txt_label:
                
                class_name = w_d.split()[0]

                if class_name not in self.object_name.keys():
                    continue

                w_d = w_d.split()[4:8] # xmin ymin xmax ymax(left top right bottom)
                w_d = list(map(float,w_d))
                bbox = self.convert_voc_to_yolo(img_shape,w_d)

                detect = [self.object_name[class_name],bbox[0],bbox[1],bbox[2],bbox[3]]
                yolo_coor.append(detect)

            out_file = f'./mAP/input/{os.path.basename(file)[:-4]}.txt'

            with open(out_file, 'w') as f:
                for item in yolo_coor:
                    item = list(map(str,item))
                    item = ' '.join(item)
                    f.write("%s\n" % item)
            if self.save_img:
                cv2.imwrite(f'./mAP/input/images-optional/{os.path.basename(file)[:-4]}.png',image)



if __name__ == "__main__":

    #################### Arguments ####################

    parser = argparse.ArgumentParser(description="convert yolo to voc format")
    parser.add_argument('--l_folder', nargs='?',default = './yolov5/runs/detect/wbf_sn',
                            help='wbf label folder')
    parser.add_argument('--im_format', type=str, default='.png',
                            help='image format (ex .png, .jpg ...)')
    parser.add_argument('--save_img', type=bool, default=True,
                            help='save images')
    args = parser.parse_args()

    l_folder = args.l_folder
    im_format = args.im_format
    save_img = args.save_img

    total_label = list(glob(l_folder+'/*.txt'))

    make_voc(total_label,im_format,save_img).main()