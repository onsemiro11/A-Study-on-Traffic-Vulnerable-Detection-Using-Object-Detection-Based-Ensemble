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

종종 긴 횡단보도나 교차로를 신호등 시간 내에 이동하지 못하시거나, 이동하시는데 위험한 상황을 격으시는 교통약자분들을 본 적이 있었다. 이런 경험을 없애고 교통약자분들도 안전한 야외활동을 할 수 있도록 도와주자는 생각으로 이번 프로젝트를 진행하였다. 처음 시작할 때의 목표는 대표적인 교통약자분들을 모두 인식하여 도움을 주자는 생각을 하였지만, 부족한 데이터로 인해 낮은 정확도를 보일 바에는 우선 많은 데이터를 가져올 수 있는 label만 가지고 모델을 구성해보자는 목표를 변경하여 진행했다.

## 프로젝트 목표

휠체어와 목발 이용자들이 횡단보도를 이동하면 카메라가 인식하여, 신호등 시간을 추가시키도록 도와주자.

<img width="1000" alt="image" src="https://github.com/onsemiro11/probono_object_detection_by_yolov5/assets/49609175/2c9ddd45-2a92-4022-9b3b-e80d08ada94f">


## 프로젝트 계획

1. 휠체어와 목발 데이터 크롤링 or 수집
2. 이미지 데이터에 인식할 물체에 라벨링 작업을 진행
3. agumentation을 진행
4. yolo v5 단일 모델 Test
5. 결과값이 아쉽다면 NMS나 WBF 앙상블 진행
6. 실제 유튜브 영상에서도 잘 인식하는지, 그리고 webcam에서도 인식을 잘하는 지 확인하고 비교 진행
7. Benchmark 모델로는 RetinaNet 모델 활용

## DATA Crawling & 수집

google 이미지에서 휠체어와 목발 사진을 크롤링하여 데이터를 수집하였다.
roboflow와 Mobility Aids 데이터를 추가로 수집
크롤링에는 selenium 을 활용하여 진행했다.

## DATA Labeling

데이터 라벨링은 roboflow.com에서 진행하였다.
label은 1을 휠체어 0을 목발로 지정하고 라벨링했다.
https://roboflow.com

수집된 데이터는 총 3,543장의 이미지 데이터셋을 구축하였다. 각 객체별 비율은 목발이용자 33.3%, 휠체어 이용자 33.2% 그리고 보행자 33.5%로 구성하여 데이터를 구축하였다.

## Data Augmentation

data Augmentation을 진행한 이유는 부족한 데이터의 문제점을 최대한 극복하기 위함이다. 기존에 갖고 있는 이미지 데이터에 affine 또는 Brightness , Blur 등을 적용하여 색다른 이미지를 제작함으로써 데이터를 증강했다.
 
기존 수집된 3,543장의 이미지 데이터 셋에서 Albumentations 패키지를 활용하여 최종 6,017장의 이미지 데이터를 확보했다. 

## YOLO V5 model

git 에서 yolo v5 을 clone한다.

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

<img width="911" alt="image" src="https://github.com/onsemiro11/probono_object_detection_by_yolov5/assets/49609175/da0a0db8-1996-4c01-8fb3-4fa27bcf3a38">

### Yolo V5 result

<img width="1000" alt="image" src="https://github.com/onsemiro11/A-Study-on-Traffic-Vulnerable-Detection-Using-Object-Detection-Based-Esemble/assets/49609175/24601cfe-a22c-428d-98d7-4f39be097efd">



## Conclusions

> 본 논문에서는 다양한 환경 속 보행자, 휠체어 사용자 그리고 목발 이용자 이미지 데이터를 사용하면서, 횡단보도속 역동적인 환경의 변화에서 검출할 수 있도록 했다. 부족한 데이터를 해결하기 위해 증강 기법을 활용하였다. 또한 모델 YOLO v5 속 모델들을 앙상블 기법 중 NMS와 WBF를 활용하여, 단일 모델의 과적합 문제를 해결하고 안정성을 갖춘 모델을 구축하였다. 실제 횡단보도 환경 속 영상에서 단일 모델과 앙상블한 모델을 비교한 결과, 단일 모델보다 앙상블된 모델이 우수한 성능을 보여줬다는 것을 확인하였다. 연산량이 비교적 적은 모델들인 s와 n을 선택하여 과적합을 최소화하려 하였고, 더 높은 안정성을 추구하기 위해 앙상블 기법을 활용하였다. 그 결과, 단일 모델에서 감지하지 못하거나 오탐지된 객체들을 올바르게 탐지하는 모습을 보여줬다.

> 본 연구는 서비스를 제공하는 횡단보도 환경의 cctv 각도에서 바라보는 데이터를 구축하지 못하였다. 부족한 데이터를 활용하여 아쉬운 성능을 보여줄 가능성이 존재한다. 하지만, 본 연구는 충분히 휠체어 사용자와 목발 이용자를 탐지하여 자동화된 스마트 신호등을 구현하는데 활용할 수 있다는 것을 보여준다. 적합한 데이터를 구축하고 높은 연산량을 가진 모델을 활용하여, 좀 더 우수한 성능을 갖춘 모델을 개발하는 연구가 필요하다. 해당 연구가 지속되면 향후 보다 높은 성능과 안정성을 보여줄 수 있을 것으로 기대된다. 나아가, 기존 연구인 [5]과 [6]의 실험 결과와 함께 교통약자에 적합한 보행 제한 시간을 적용하여 자동으로 변동 시간을 제공하는 신호등 개발에 기여할 수 있다.
