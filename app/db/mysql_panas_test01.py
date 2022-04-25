import pymysql
import pandas as pd

host = 'localhost'
user = 'root'
password = '1234'
database = 'motd_db'

# 접속하기
db = pymysql.connect(
    host = host, # 접근 주소
    # port=PORT,  # 접근 포트 번호
    user = user,  # 아이디
    passwd = password,  # 비밀번호
    db = database,  # DB 이름
    charset = 'utf8',    
    cursorclass = pymysql.cursors.DictCursor,
    init_command='SET NAMES UTF8'   # UTF8 로  가져오기
)

'''
# DB 명칭 조회하기

cursor = db.cursor()   
sql = "select * from info limit 5"
cursor.execute(sql)
result = cursor.fetchall()
print(result)
'''

# 특정 TABLE 데이터 가져오기
TABLE_NAME = 'main_test'  # table명
cursor = db.cursor()    
# 디비 가져오기
sql = f"SELECT * FROM {TABLE_NAME}"
cursor.execute(sql)
table_data = cursor.fetchall() 
# print(table_data) 

# SQL 서버에서 받은 데이터 판다스로 정리하기

import pandas as pd

df = pd.DataFrame(table_data)
# print(df.head(5))

# df.to_csv("./df_info.csv")

df1 = df.set_index('id')
print(df1.head(5))