from PIL import ImageColor
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def yolo_to_xml_bbox(bbox, image): #위 좌표 변환 함수가 잘 된 것인지 다시 변환하여 이미지에 mapping해보기위해 함수를 따로 만든 것.
    # x_center, y_center width, heigth
    h, w, c = image.shape
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return (xmin, xmax, ymin, ymax)



def plot_box(image, top_left_point, bottom_right_point, width, height, label, color=(100,50,250), padding=6, font_scale=0.9):
    label = label.upper()
    thickness = int((image.shape[0] + image.shape[1]) / 200)  # Adjust this multiplier as needed

    cv2.rectangle(image, (top_left_point['x'] - 1, top_left_point['y']), (bottom_right_point['x'], bottom_right_point['y']), color, thickness=thickness)
    res_scale = (image.shape[0] + image.shape[1])/1600
    font_scale = font_scale * res_scale
    font_width, font_height = 0, 0
    font_face = cv2.FONT_ITALIC
    text_size = cv2.getTextSize(label, font_face, fontScale=font_scale, thickness=1)[0]

    if text_size[0] > font_width:
        font_width = text_size[0]
    if text_size[1] > font_height:
        font_height = text_size[1]
    if top_left_point['x'] - 1 < 0:
        top_left_point['x'] = 1
    if top_left_point['x'] + font_width + padding*2 > image.shape[1]:
        top_left_point['x'] = image.shape[1] - font_width - padding*2
    if top_left_point['y'] - font_height - padding*2  < 0:
        top_left_point['y'] = font_height + padding*2
    
    p3 = top_left_point['x'] + font_width + padding*2, top_left_point['y'] - font_height - padding*2
    cv2.rectangle(image, (top_left_point['x'] - 2, top_left_point['y']), p3, color, -1)
    x = top_left_point['x'] + padding
    y = top_left_point['y'] - padding
    cv2.putText(image, label, (x, y), font_face, font_scale, [0, 0, 0], thickness=2)

    return image


if __name__ == "__main__":

    #################### Arguments ####################

    parser = argparse.ArgumentParser(description="draw_box")
    parser.add_argument('--im_path', nargs='?',default = './data/images/data_0.png',
                            help='image path')
    args = parser.parse_args()

    image_path = args.im_path
    txt_path = f'data/labels/{os.path.basename(image_path)[:-4]}.txt'

    class_name = {0:'crutches',1:'wheelchair',2:'pedestrian'}
    #if you wanna diffrent color of each box, use this color dic.
    # color = {'crutches':(235,75,66),
    #         'pedestrian':(236,120,55),
    #         'wheelchair':(238,160,145)}

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_index = float(line[0])
            x_center, y_center, width, height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            xmin, xmax, ymin, ymax = yolo_to_xml_bbox([x_center, y_center, width, height], image)
            print(xmin, xmax, ymin, ymax)
            image = plot_box(image, {'x':xmin, 'y':ymin}, {'x':xmax, 'y':ymax}, width, height, class_name[int(class_index)],color =(255,228,30)) #chage color = color
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()
