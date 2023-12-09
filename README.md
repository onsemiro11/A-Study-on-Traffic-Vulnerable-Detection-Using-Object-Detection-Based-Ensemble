## :runner:팀원

<table>
  <tr>
    <td align="center"><a href="https://github.com/onsemiro11"><img src="https://avatars.githubusercontent.com/u/49609175?v=4" width="200px;" alt="teammember2"/><br /><h3><b><a href="https://github.com/onsemiro11">onsemiro11</b></h3></a><br /></td>
    <td align="center"><a href="https://github.com/ksun0401"><img src="https://avatars.githubusercontent.com/u/70461025?v=4" width="200px;" alt="teammember1"/><br /><h3><b><a href="https://github.com/ksun0401">coby</b></h3></a><br /></td>
    <td align="center"><a href="https://github.com/Naseungchae"><img src="https://avatars.githubusercontent.com/u/90239125?v=4" width="200px;" alt="teammember2"/><br /><h3><b><a href="https://github.com/Naseungchae">Naseungchae</b></h3></a><br /></td>
    <td align="center"><a href="https://github.com/YUL-git"><img src="https://avatars.githubusercontent.com/u/89930713?v=4" width="200px;" alt="teammember2"/><br /><h3><b><a href="https://github.com/YUL-git">YUL</b></h3></a><br /></td>
  </tr>
<table>


# A Study on Traffic Vulnerable Detection Using Object Detection-Based Esemble

Despite ongoing attempts to address pedestrian accidents at crosswalks, they continue to occur. In particular, the elderly, the disabled, and other vulnerable groups are at greater risk of being involved in traffic accidents. It is necessary to look at the problem carefully. This paper proposes an object detection technology using the YOLO v5 model for pedestrians using assistive devices such as wheelchairs and crutches. Image crawling, Roboflow and Mobility Aids data of wheelchair and crutch users and pedestrians were collected. Data augmentation techniques were used to improve the generalization performance. In addition, ensemble techniques were used to reduce type 2 errors, resulting in a high performance figure of 96% recall. This proves that ensembling a single model in YOLO to target the transportation disadvantaged can provide accurate detection performance without missing objects.

## Paper

soon....

## Project Work Flow

![image](https://github.com/onsemiro11/A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble/assets/49609175/eecadfb0-d501-434f-888d-989488acda11)

```
A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble
    |-- data
        |-- data.yaml
        |images
            |-- data_0.png
            |-- ...
        |labels
            |-- data_0.txt
            |-- ...
        |-- re_train.csv
        |-- re_test.csv
        |-- re_class.csv
    |-- mAP
        |detection-result
        |input 
        |output
        |-- main.py
    |-- pytorch-retinanet # clone github
        |data
        |images #Test data
        |retinanet
        |-- model_final.pt
        |-- train.py
        |-- visualize_single_image.py
        |-- visualize.py
        |-- csv_valibation.py
    |-- yolov5 # clone github
        |models
        |utils
        |-- detect.py
        |-- train.py
        |-- export.py
        |-- val.py
    |-- requirements.txt
    |-- crawling.py
    |-- augmentation.py
    |-- draw_box.py
    |-- ensemble_wbf.py
    |-- voc2yolo.py
    |-- yolo2voc.py
    |-- retinanet_csv.py 
```

## Data

- Google Crawling
  ```shell
     python3 crawling.py
  ```
- Mobibity Aids : http://mobility-aids.informatik.uni-freiburg.de/
- Roboflow

This is our Pre-augmented DataSet Link

> Google Drive Link : https://drive.google.com/file/d/1eRDazUBCUFZIwlLjkNzSV6Nqfh-Ita9H/view?usp=drive_link

> Roboflow Link :

### Data augmentation

3543 images -> 6017 images

augmentation : Brightness, flip left and right, and rotate within 30 degrees.

```shell
   python3 augmentation.py
```

### Drawing boxes on a data image

- Converting Voc format to yolo format.
```shell
   python3 voc2yolo.py --l_folder <label_txt_folder> --im_format .png
```
- Drawing
```shell
   python3 draw_box.py --im_path <image_path>
```


## Install

Clone repo and install requirements.txt

```shell
  git clone https://github.com/onsemiro11/A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble.git  # clone
  cd A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble
  pip install -r requirements.txt  # install
```

## Training

We trained YOLO v5. And selected Retinanet as the benchmark model for comparison.

### YOLO V5 Training

You can use Yaml file(.data/data.yaml) that shows data information and dataset constitution.

- Install & Training
```shell
  git clone https://github.com/ultralytics/yolov5  # clone
  cd yolov5
  # yolov5 s model
  python3 train.py --img 418 --batch 32 --epochs 50 --data A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble/data/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5s_results
  # yolov5 n model
  python3 train.py --img 418 --batch 32 --epochs 50 --data A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble/data/data.yaml --cfg ./models/yolov5n.yaml --weights yolov5n.pt --name yolov5n_results
```
- Inference
```shell
  cd yolov5
  python detect.py --weights ./runs/train/yolov5s_results/weights/best.pt --img 418 --source '<clone name>/data/val.txt' --save-txt --save-conf
  python detect.py --weights ./runs/train/yolov5n_results/weights/best.pt --img 418 --source '<clone name>/data/val.txt' --save-txt --save-conf
```

### RetinaNet (Benchmark model) Training

- make a data csv file
```shell
   python3 retinanet_csv.py --l_folder <label_folder_path> --im_format .png --split_d 0.3
```
- Install & Clone
```shell
  apt-get install tk-dev python-tk
  git clone https://github.com/yhenon/pytorch-retinanet.git
```
- Training
```shell
  cd pytorch-retinanet
  python3 --dataset csv --csv_train <rt_train.csv_path>  --csv_classes <rt_class.csv_path>  --csv_val <rt_val.csv_path> --epochs 50
```
- Inference
```shell
  cd pytorch-retinanet
  python3 visualize_single_image.py --image_dir "./images" --model_path "./model_final.pt" --class_list <rt_class.csv_path>
```

## Ensemble model

Ensemble model : yolov5 s model & yolov5 n model

### NMS (Non-Maximum-suppression)

- validataion Test
```shell
  cd yolov5
  python3 val.py --weights yolov5s.pt yolov5n.pt --data <clone_name>/data/data.yaml --img 418 --half
```

- Detection
```shell
  cd yolov5
  python3 detect.py --weights yolov5s.pt yolov5n.pt --img 418 --source <test_images_folders>
```

### WBF (Weighted Boxes Fusion)

- wbf ensemble (you can change image path, txt path, out path, iou threshold , skip box threshold in ensemble_wbf.py)
```shell
  python3 ensemble_wbf.py
```

- Calculate index of evaluation(mAP , Precision , Recall)

  please , refer to https://github.com/Cartucho/mAP

  1. Setting up detection-results , ground-truth , images-optional folders ( convert yolo format to voc format )
  ```shell
  # set up detection-results folder & images-optional folders
  python3 yolo2voc.py
  
  # set up ground-truth folder
  python3 yolo2voc.py --l_folder ./data/label --save_img False
  ```
  2. Calculate 
  ```shell
    cd mAP
    python3 main.py
  ```

## Model Result

|Model|mAP @0.5|Recall|
|:------:|:---:|:---:|
|RetinaNet|0.90|0.88|
|Yolo v5 s|0.94|0.91|
|Yolo v5 n|0.91|0.87|
|Yolo v5 n&s wbf|0.93|0.96|
|Yolo v5 n&s nms|0.95|0.92|

## Conclusions

In this study, we aimed to detect pedestrians, wheelchair users, and crutch users in various environments in dynamic crosswalks. In this process, we utilized image augmentation techniques to improve learning accuracy. To construct the model, we used NMS and WBF among the ensemble techniques of YOLO v5 models to build a stable model that can solve the overfitting problem of a single model and minimize the type 2 error. When comparing the single model and the ensemble model in the actual crosswalk environment, the ensemble model minimizes the computation and satisfies the purpose of this study by combining small models. We tried to minimize overfitting through the selection of low-computation models, s and n, and the augmentation technique, and used the ensemble technique to pursue higher recall performance. As a result, we were able to correctly detect objects that were not detected or falsely detected by a single model.
The limitations of this study are as follows First, we were unable to utilize the CCTV angle data of the crosswalk environment we serve. A more sophisticated model could be built by utilizing data from different angles. However, since most CCTVs collect data from a fixed location, the proposed model in this study is meaningful. In addition, future research on traffic vulnerability detection is expected to develop a traffic system that automatically provides appropriate walking time in conjunction with the traffic signal system.
