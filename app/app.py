# =========== <import> ==================================

import numpy as np
import db
from tensorflow.python.keras.models import load_model
from flask import Flask, render_template, request, jsonify, json
from PIL import Image
from keras.layers import BatchNormalization
from flask_cors import CORS
import os
# =========== <Flask 객체 app 생성 및 설정(Json, Ascii, Corpse)> =====================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)  # cors 설정 -> 교차검증
UPLOAD_FOLD = r'.\UPLOAD_FOLDER' 
app.config['UPLOAD_FOLDER'] = r'.\UPLOAD_FOLDER'
# ========== <딥러닝 모델 호출(?)> ===================================================

global model
model = load_model('47-0.310246.h5',
                   custom_objects={'BatchNormalization': BatchNormalization})


# ============ << 이하 Back 메인 로직 >> ===================


@app.route('/', methods=['POST'])
def inference():
    # images로 넘어온게 없다면 files에 있는지 체크
    if 'images' not in request.files:
        return 'images is missing', 777

    # request에 파일 받기 request.files['key_name']
    img = request.files['images']
    path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(path)
    f = open(path,'rb')
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((200, 200))
    data2 = np.asarray(img)
    data2 = data2.astype('float') / 255
    X.append(data2)
    X = np.array(X)

    # 예측
    pred = model.predict(X)
    pred = np.round(pred, 2)

    return(res(pred))


# ==== 결과 전달을 위한 카테고리, 멘트 List 생성, categories.py 참조 ==========

categories = [
    "직장인💼", "뉴요커📸", "바캉스⛱", "러블리🌷", 
    "GD😎", "힙스터🏃", "불금🔥", "대학생🙋"]

'''
categories = [
    "직장인", "뉴요커", "바캉스", "러블리", 
    "GD", "힙스터", "불금", "대학생"]
'''

motd_mention = [

    "깔끔하고 엣지있게, 칼퇴하고픈 여의도 직장인 무드💼", 
    "패션의 끝은 역시 캐주얼 꾸안꾸! 시크 뉴요커 무드📸", 
    "여긴 혹시.. 해변?! 시원한 하와이 바캉스 무드⛱",
    "하늘하늘 사랑스러운 데이트룩! 큐티 러블리 무드🌷",
    "누구보다 화려하게 남들과는 다르게, 내가 바로 GD😎", 
    "가볍고 활기차게, 한강 소풍하는 스포티 힙스터 무드🏃",
    "나에게 숨겨진 섹시가? 핫한 오늘의 motd는 불금 무드🔥",
    "나랑 같이 산책할래? 캐주얼 캠퍼스룩, 새내기 대학생 무드🙋"

]

# ===========================================================================

# ==== 예측값을 받아서 카테고리랑 매칭 한 뒤에 json형식으로 바꿔서 리턴 =====

result = {}
mood = []

def res(pred):
    id = request.form["data"]
    id2 = json.loads(id)
    
    # int형 결과 리스트 [[], [], []...]
    a = []
    for i in range(len(categories)):
        temp = []
        b = (categories[i])

        temp.append(b)
        temp.append(motd_mention[i])
        temp.append(int(np.round(pred[0][i]*100, 2)))
        
        a.append(temp)

    print(a, '\n')

    # 정렬
    for i in range(len(a)-1):
        max_idx = i
        for j in range(i+1, len(a)):  
            if a[max_idx][2] < a[j][2]:
             max_idx = j 
        a[i], a[max_idx] = a[max_idx], a[i]

    print(a, '\n')

    # int -> str
    for i in range(len(a)): 
        a[i][2] = str(a[i][2])

    print(a, '\n')
    mood = a

    result["id"] = id2["id"]
    result['mood'] = a

    # result_style 추출
    result_style = a[0][0]
    print(result_style)
    print(type(result_style))
    result_style = result_style[:-1]

    toDB(result_style)  # toDB() : DB 테이블에 insert하는 함수    ?????????
    return jsonify(result)


# =========== DB 테이블에 결과 입력 ============================


def toDB(result_style):
    data = request.form["data"]
    my_data = json.loads(data)

    gender = my_data['gender']
    age = my_data["age"]
    # result_style = "섹시"

    print()
    print(gender)
    print(type(gender))
    print("age : ", age)
    print(result_style)
    
    db.insert(gender, age, result_style)


# ============ Dash + HTML : 정적인 데시보드(그래프) ==============
# ========     결과를 html로 저장하여 뿌리는 형태 -> 최악의 경우 고려

@app.route('/dash_style')
def dash_style():
    return render_template('dash_style.html')


@app.route('/dash_gender')
def dash_gender():
    return render_template('dash_gender.html')


@app.route('/dash_age')
def dash_age():
    return render_template('dash_age.html')


# ============= HTML TEST ====================


@app.route('/html01')
def html01():
    return render_template('index01.html')


# ============= 서버 실행 ====================
#         Port : 3216

if __name__ == '__main__':
    app.run()  
# []