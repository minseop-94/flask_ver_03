import pymysql


# ========= 수정 필요!!! ============

host = 'localhost'
user = 'root'
password = '1234'
database = 'motd_db'


# ========= <insert : DB 테이블에 데이터 입력> ============


def insert(gender, age, result_style):

    data = [gender, age, result_style]
    try:
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     cursorclass=pymysql.cursors.DictCursor)
        with connection:
            with connection.cursor() as cursor:
                # 쿼리중 오류가 나더라도, 커넥션은 정상적으로 닫아야 함으로 예외처리 추가
                try:
                    sql = '''
                        INSERT INTO `motd_db`.`main_test` (`gender`, `age`, `result_style`) 
                        VALUES (%s, %s, %s)
                    '''
                    cursor.execute(sql, data)
                    connection.commit()
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)



# ========= <select : 결과 조회> : 시각화 할때 필요 ============


def selectAll():    
    result = None
    try:
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     cursorclass=pymysql.cursors.DictCursor)
        with connection:
            with connection.cursor() as cursor:
                # 쿼리중 오류가 나더라도, 커넥션은 정상적으로 닫기 위해 예외처리 추가
                try:
                    sql = '''
                        select * from main_test
                    '''
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    # print(result)
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
    return result




# =====================================

if __name__ == '__main__':
    # select_login('guest', '1')
    print('이 파일을 실행함')
else:
    print('다른 모듈이 가져다가 사용시 호출')
    pass