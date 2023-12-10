from PIL import Image
from tqdm import tqdm
from ensemble_boxes import *

def xywh2x1y1x2y2(bbox):
    x1 = bbox[0] - bbox[2]/2
    x2 = bbox[0] + bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    y2 = bbox[1] + bbox[3]/2
    return ([x1,y1,x2,y2])

def x1y1x2y22xywh(bbox):
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return ([x,y,w,h])

IMG_PATH = '.data/images/'
TXT_PATH = './yolov5/runs/detect/exp0/'

OUT_PATH = './yolov5/runs/detect/wbf_sn/'


MODEL_NAME = os.listdir(TXT_PATH)

# ===============================
# Default WBF config (you can change these)
iou_thr = 0.55
skip_box_thr = 0.01
# skip_box_thr = 0.0001
sigma = 0.01
# boxes_list, scores_list, labels_list, weights=weights,
# ===============================

image_id = open('.data/val.txt','r')
image_ids = image_id.readlines()

for image_id in tqdm(image_ids, total=len(image_ids)):
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    for name in MODEL_NAME:
        # print(name)
        box_list = []
        score_list = []
        label_list = []
        txt_file = TXT_PATH +name+'/'+ image_id.split('/')[-1].replace('jpg', 'txt').replace('\n','')
        # print(txt_file)
        if os.path.isfile(txt_file):
            txt_df = pd.read_csv(txt_file,header=None,sep=' ').values
            # print(txt_df)
            for row in txt_df:
                box_list.append(xywh2x1y1x2y2(row[1:5]))
                score_list.append(row[5])
                label_list.append(int(row[0]))
            boxes_list.append(box_list)
            scores_list.append(score_list)
            labels_list.append(label_list)
            weights.append(1.0)
        else:
            continue


    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    out_file = open(OUT_PATH + image_id.split('/')[-1].replace('jpg', 'txt').replace('\n',''), 'w')

    for i,row in enumerate(boxes):
        img = Image.open(image_id.replace('\n',''))
        img_size = img.size
        bbox = x1y1x2y22xywh(row)
        out_file.write(str(int(labels[i])) + ' ' +" ".join(str(x) for x in bbox) + " " + str(round(scores[i],6)) + '\n')
    out_file.close()

# from ensemble_boxes import *
# for image_id in tqdm(df['image_id'], leave=False):
#         image_df = df[df['image_id']==image_id].reset_index(drop=True)
#         h, w = image_df.loc[0, ['height', 'width']].values
#         boxes = image_df[['x_min', 'y_min',
#                           'x_max', 'y_max']].values.tolist()
#         # Normalise all the bounding boxes (by dividing them  by size-1
#         boxes = [[j/(size-1) for j in i] for i in boxes]
#         scores = [1.0]*len(boxes) # set all of the scores to 1 since we only have 1 model here
#         labels = [float(i) for i in image_df['class_id'].values]
#         boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels],weights=None,iou_thr=iou_thr,
#                                           skip_box_thr=skip_box_thr)
#         list_image.extend([image_id]*len(boxes))
#         list_h.extend([h]*len(boxes))
#         list_w.extend([w]*len(boxes))
#         list_boxes.extend(boxes)
#         list_cls.extend(labels.tolist())
#     # bring the bounding boxes back to their original size  (by multiplying by size - 1)
#     list_boxes = [[int(j*(size-1)) for j in i] for i in list_boxes]
#     new_df['image_id'] = list_image
#     new_df['class_id'] = list_cls
#     new_df['h'] = list_h
#     new_df['w'] = list_w
#     # Unpack the coordinates from the  bounding boxes
#     new_df['x_min'], new_df['y_min'], \
#     new_df['x_max'], new_df['y_max'] = np.transpose(list_boxes)