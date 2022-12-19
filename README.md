# 교통약자_object_detection_by_yolov5

종종 긴 횡단보도나 교차로를 신호등 시간 내에 이동하지 못하시거나, 이동하시는데 위험한 상황을 격으시는 교통약자분들을 본 적이 있었다. 이런 경험을 없애고 교통약자분들도 안전한 야외활동을 할 수 있도록 도와주자는 생각으로 이번 프로젝트를 진행하였다. 처음 시작할 때의 목표는 대표적인 교통약자분들을 모두 인식하여 도움을 주자는 생각을 하였지만, 부족한 데이터로 인해 낮은 정확도를 보일 바에는 우선 많은 데이터를 가져올 수 있는 label만 가지고 모델을 구성해보자는 목표를 변경하여 진행했다.

## 프로젝트 목표

휠체어와 목발 이용자들이 횡단보도를 이동하면 카메라가 인식하여, 신호등 시간을 추가시키도록 도와주자.

## 프로젝트 계획

1. 휠체어와 목발 데이터를 google에서 크롤링한다.
2. 이미지 데이터에 인식할 물체에 라벨링 작업을 진행하였다.
3. 크롤링한 데이터로 부족할 것이 분명하니, agumentation을 진행한다.
4. yolo v5 모델에 데이터를 넣어 결과를 확인해 본다.
5. 결과값이 아쉽다면, 데이터를 다시 정제하거나, 모델 파라미터를 바꾼 후 진행해본다.(반복)
6. 실제 유튜브 영상에서도 잘 인식하는지, 그리고 webcam에서도 인식을 잘하는 지 확인해본다.

## DATA Crawling

google 이미지에서 휠체어와 목발 사진을 크롤링하여 데이터를 수집하였다.
크롤링에는 selenium 을 활용하여 진행했다.

## DATA Labeling

데이터 라벨링은 roboflow.com에서 진행하였다.
label은 1을 휠체어 0을 목발로 지정하고 라벨링했다.
https://roboflow.com

## Data Augmentation

data Augmentation을 진행한 이유는 부족한 데이터의 문제점을 최대한 극복하기 위함이다. 기존에 갖고 있는 이미지 데이터에 affine 또는 Brightness , Blur 등을 적용하여 색다른 이미지를 제작함으로써 데이터를 증강했다.
 
활용한 패키지는 albumentations다.

## YOLO V5 model

git 에서 yolo v5 을 불러온다.

requirements.txt에 있는 분석에 필요한 패키지들도 모두 pip install을 통해 저장한다.

### yaml 파일 제작

학습할 데이터의 경로, 클래스 갯수 그리고 종류를 적어 놓는 yaml 파일을 제작해야한다.
- train : 학습 데이터 폴더 경로
- val : 검증 데이터 폴더 경로
- nc : 학습할 클래스 갯수
- names : 학습할 클래스 이름들

### Yolo v5 Dataset 불러오기
이전 글에서 labeling하고 augmentation한 이미지와 좌표(txt파일)을 불러와서 각 해당 변수에 할당 시키고
둘의 파일 개수가 동일한지 확인해봤다.

### Yolo v5 train_test data 분류

우선 validation과 train 데이터를 3:7로 분리해줬다.
그 후, validation에서 5:5로 validation과 test데이터를 나눴다.
 
validation을 따로 생성한 이유는,
- 최대한 test데이터를 건들지 않고 validation데이터를 활용해서 학습한 후, test 데이터로 평가해보기 위해서다.

### Yolo v5 modeling

Yolo model의 원리
- 예측할 이미지를 grid size로 나눠주고, 나눠진 이미지들의 각 confidence score들을 계산해가면서,
기존 boundary boxes와 비슷한지와, 객체가 존재할 가능성을 convolution layer를 계속해서 통과하면서 loss값이 최소화되는 부분을 추출하는 model이다.
 
모델을 돌릴때는, image size와 batch size, epochs 크기, data 경로, Configuration yaml 파일, weights를 저장할 파일, result를 저장할 곳을 지정해서 입력해준다.
![image](https://user-images.githubusercontent.com/49609175/208351421-2697e55e-869b-4efb-bb20-b67865818310.png)

위와 같은 결괏값이 나온다.
 
휠체어의 precision과 recall 값이 조금 부족하다는 것을 확인 할 수 있다.
mAP@.5는 IoU값이 0.5 이상인 것들의 mAP값을 말하는 것이고,
mAO@.5:.95는 IoU값이 0.5부터 0.95 사이인 것들의 mAP값을 말한다.
 
조금 부족한 결괏값을 보여준다.
데이터를 다시 수집하고 증강시켜서 모델을 돌려봤다.

![image](https://user-images.githubusercontent.com/49609175/208351457-5e367a5b-24c4-4b47-8da5-45ba96ba2f6b.png)



### Yolo V5 result

![image](https://user-images.githubusercontent.com/49609175/208351109-19f0bbd1-edec-4861-9151-8807a9c661c4.jpeg)



## 프로젝트 보완점

> result image에서 보았듯이 아직 detection되지 않은 객체들이 많다. 좀더 정확도를 올리기 위해, 다양한 데이터 수집과 정교한 전처리가 필요할 것같다.
