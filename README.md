# learning-ai-knowledge

[develop] <br>
└──[study/yolov5_tutorial] <br>
    ├── <br>
    │ <br>
    │ <br>

### official github for yolov5
https://github.com/ultralytics/yolov5


### prerequisite
~~~
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
~~~
### 
~~~
### official example
python3 detect.py --weight yolov5s.pt --source  https://www.youtube.com/watch?v=Zgi9g1ksQHc
### 운전 시내-뉴욕시 -미국
python3 detect.py --weight yolov5s.pt --source https://www.youtube.com/watch?v=7HaJArMDKgI
###  서울 강남의 폭우 내리는 밤
python3 detect.py --weight yolov5s.pt --source https://www.youtube.com/watch?v=0UlOohkMZnI
### 강남 압구정동, 압구정로데오의 밤거리
python3 detect.py --weight yolov5s.pt --source https://www.youtube.com/watch?v=la7J3PLAXGg
~~~

참고하면 좋은 사이트<br>
https://yhkim4504.tistory.com/

coco dataset
https://cocodataset.org/#download


curl -L "https://universe.roboflow.com/ds/qcv8WsT2ts?key=BDJ1Feb6ss" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip