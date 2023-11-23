import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import cv2
from sklearn.model_selection import train_test_split


class make_retinadata():

  def __init__(self,file_name, im_format):
    self.file_name = file_name
    self.im_format = im_format
    self.object_name = {'0':'crutches','1':'wheelchair','2':'pedestrian'}


  def convert_yolo_to_voc(self,x_c_n, y_c_n, width_n, height_n, img_width, img_height):
      ## remove normalization given the size of the image
      x_c = float(x_c_n) * img_width
      y_c = float(y_c_n) * img_height
      width = float(width_n) * img_width
      height = float(height_n) * img_height
      ## compute half width and half height
      half_width = width / 2
      half_height = height / 2
      ## compute left, top, right, bottom
      ## in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
      left = int(x_c - half_width) + 1
      top = int(y_c - half_height) + 1
      right = int(x_c + half_width) + 1
      bottom = int(y_c + half_height) + 1
      return left, top, right, bottom


  def main(self):

    retina_list = []

    for file in tqdm(self.file_name):
      txt_file = open(file,'r')
      txt_label = txt_file.readlines()
      
      img_path = file.replace('.txt',self.im_format).replace('labels','images')
      img = cv2.imread(img_path)

      for w_d in txt_label:
        w_d = w_d.split()
        bbox = self.convert_yolo_to_voc(w_d[1],w_d[2],w_d[3],w_d[4],img.shape[1],img.shape[0])

        detect = [img_path,bbox[0],bbox[1],bbox[2],bbox[3],self.object_name[str(int(float(w_d[0])))]]
        retina_list.append(detect)

    return retina_list
  

if __name__ == "__main__":

  #################### Arguments ####################

  parser = argparse.ArgumentParser(description="make_retinadata")
  parser.add_argument('--l_folder', nargs='?',default = './data/labels',
                          help='labels folder path except last /')
  parser.add_argument('--im_format', type=str, default='.png',
                          help='image format (ex .png, .jpg ...)')
  parser.add_argument('--split_d', type=float, default=0.3,
                          help='train validation data split degree')
  args = parser.parse_args()

  l_folder = args.l_folder
  im_format = args.im_format
  split_d = args.split_d

  total_label = glob(l_folder+'/*.txt')

  train_data, val_data = train_test_split(total_label, test_size=split_d, random_state=43)

  # split & make train csv data
  train_list = make_retinadata(train_data, im_format)
  train_final = train_list.main()

  # split & make validation csv data
  val_list = make_retinadata(val_data,im_format)
  val_final = val_list.main()

  #save train data csv
  retina_train = pd.DataFrame(train_final)
  print(retina_train[:10])
  print('data/rt_train.csv',' : ', len(retina_train))
  retina_train.to_csv('data/rt_train.csv',header=None,index=False)

  #save validation data csv
  retina_val = pd.DataFrame(val_final)
  print(retina_val[:10])
  print('data/rt_val.csv',' : ', len(retina_val))
  retina_val.to_csv('data/rt_val.csv',header=None,index=False)

  #make class csv
  class_data = [['crutches','0'],['wheelchair','1'],['pedestrian','2']]
  retina_class = pd.DataFrame(class_data)
  retina_class.to_csv('data/rt_class.csv',header=None,index=False)

