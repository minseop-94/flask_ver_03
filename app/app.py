
# =========== import ==================================

import numpy as np
import db
# import tensorflow as tf
from tensorflow.python.keras.models import load_model
from flask import Flask, render_template, request, jsonify, json
from PIL import Image
# import os, glob, time
from keras.layers import BatchNormalization
from flask_cors import CORS
# import cv2 
import os
# =========== Flask 객체 app 생성 및 설정(Json, Ascii, Corpse) ==================================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# cors 설정 -> 교차검증
CORS(app)

# ========== 딥러닝 모델 호출(?) ===================

global model
model = load_model('47-0.310246.h5',
                   custom_objects={'BatchNormalization': BatchNormalization})


UPLOAD_FOLD = r'.\UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = r'.\UPLOAD_FOLDER'


categories = ["무드1-직장인", "무드2-캐주얼", \
                "무드3-리조트",  "무드4-데이트", \
                "무드5-패턴",   "무드6-스포티", \
                "무드7-섹시",   "무드8-캠퍼스"]

motd_mention = [
    "깔끔하고 엣지있게! 여의도 직장인무드💼",
    "패션의 끝은 역시 캐주얼 꾸안꾸!, 뉴요커 무드😎",
    "여긴 혹시.. 해변 ?! 시원한 바캉스룩, LA해변 바캉스 무드🏖",
    "하늘하늘 사랑스러운 데이트룩, 러블리 무드🌷",
    "누구보다 화려하게 남들과는 다르게, 개성 무드😎",
    "가볍고 활기차게, 오늘은 스포티 무드!🏃",
    "핫한 오늘의 ootd는 불금무드!🔥",
    "캐주얼 캠퍼스룩! 오늘은 대학생 무드🙋 ♂️",
  ]

# ========== Back 메인 로직 ===================


@app.route('/', methods=['POST'])
def inference():
    # images로 넘어온게 없다면 files에 있는지 체크
    if 'images' not in request.files:

        return 'images is missing', 777

    # request에서 파일 받기 request.files['key_name']
    # img = request.files['images']

    # “python filestorage object to image”
    
    # read image file string data
    # imgstr = img.read()
    # convert string data to numpy array
    # npimg = np.fromstring(imgstr, np.uint8)
    # convert numpy array to image 원래 참고
    # img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
    # img = cv2.imdecode(np.fromstring(imgstr, np.uint8), cv2.IMREAD_UNCHANGED) //원본코드
    # img = cv2.imread(imgstr, cv2.IMREAD_UNCHANGED)
    # 이미지 resize, 변환 후
    X = []

    # img = Image.fromarray(npimg.astype('uint8')) // 잠시 image 테스트를 위해 
    img = request.files['images']
    path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(path)
    # print(path)
    # print("여기서 시작인가? img.convert()들어가기 직전 img: ", img)

    
    f = open(path,'rb')
    img = Image.open(f)
    img = img.convert("RGB")
    
    # print("img -> rgb 여기까지 test 2 옴")

    img = img.resize((200, 200))
    data2 = np.asarray(img)
    data2 = data2.astype('float') / 255
    X.append(data2)
    X = np.array(X)

    # 예측
    pred = model.predict(X)
    pred = np.round(pred, 2)

    # print(pred[0][0])
    # print(categories[0]+':'+str(pred[0][0]))
    # print(categories[1]+':'+str(pred[0][1]))
    # print(categories[2]+':'+str(pred[0][2]))
    # print(categories[3]+':'+str(pred[0][3]))
    # print(categories[4]+':'+str(pred[0][4]))
    # print(categories[5]+':'+str(pred[0][5]))
    # print(categories[6]+':'+str(pred[0][6]))
    # print(categories[6]+':'+str(pred[0][7]))

    return(res(pred))


# =========== 결과 전달을 위한 List 생성, categories.py 참조 ======

# categories = ["무드1-클래식", "무드2-페미닌", "무드3-레트로",
#               "무드4-히피", "무드5-스포티", "무드6-섹시", "무드7-톰보이"]


# 🧨현재 카테고리가 프론트랑 맞아야 돌아감


# 카테고리 설명 수정 : 위 카테고리 명이랑 달라서 내가 헷갈림

###############################################################


result = {}
# 예측값을 받아서 카테고리랑 매칭 한 뒤에 json형식으로 바꿔서 리턴


def res(pred):
    # result = {}
    id = request.form["data"]
    id2 = json.loads(id)
    # id = str(id)
    a = {}
    toarr = []
    for i in range(len(categories)):  # len(categories) : 7
        temp = []
        b = (categories[i])   # 카테고리 명
        #b= b.encode('utf-8')
        #b= b.decode('unicode_escape')
        #b = b.decode('cp949').encode('utf-8')

        # 원래 코드 str
        temp.append(str(int(np.round(pred[0][i]*100, 2))))   # 예측값
        temp.append(motd_mention[i])
        # print(temp)
        # a[b]=str(int(np.round(pred[0][i]*100,2)))
        a[b] = temp
    # pre_results.append(a)
    toarr.append(a)
    result["id"] = id2["id"]
    # result['mood'] = toarr
    result['mood'] = toarr

    # print(result)
    # print()
    # print(type(result))   # dict

    # print(result.keys)
    # print()

    # 퍼센트 최대값 구하여, result_style 추출 로직
    #     -> DB 데이터 삽입에 필요
    a1 = result['mood'][0]
    num = 0
    max = int(a1['무드1-직장인'][0])
    result_style = 0
    for i in a1:
        if max < int(a1[i][0]):
            max = int(a1[i][0])
            result_style = i   # 가장 수치 높은 스타일

    # print('1등 스타일 :', result_style, max, '%')

    toDB(result_style)  # toDB() : DB 테이블에 insert하는 함수
    # return jsonify(result)
    return jsonify(result)
    # return pre_results


def toDB(result_style):
    data = request.form["data"]
    my_data = json.loads(data)

    gender = my_data['gender']
    age = my_data["age"]
    # result_style = "섹시"

    # print("gender : ",gender)
    # print(type(gender))
    # print("age : ", age)
    db.insert02(gender, age, result_style)


# ============ Dash + HTML : 정적인 데시보드(그래프) ==============
# 결과를 html로 저장하여 뿌리는 형태. -> 최악의 경우 고려

@app.route('/dash_style')
def dash_style():
    return render_template('dash_style.html')


@app.route('/dash_gender')
def dash_gender():
    return render_template('dash_gender.html')


@app.route('/dash_age')
def dash_age():
    return render_template('dash_age.html')


# ============= TEST ====================

@app.route('/html01')
def html01():
    return render_template('index01.html')


@app.route('/test')
def test():
    return 'test🤣🤣'


@app.route('/insert01')
def insert01():
    db.insert01()
    return "please🤣"


@app.route('/insert02')
def insert02():    # 성공

    gender = "남자"
    age = "10대"
    result_style = "패턴"

    db.insert02(gender, age, result_style)
    return '메세지 : 🤣insert02'


# ============= 서버 실행 ====================

if __name__ == '__main__':
    app.run()
