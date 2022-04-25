import pymysql

host = 'first-db-busan-final.csxhjhypihxx.ap-northeast-2.rds.amazonaws.com'
user = 'root'
password = '12341234'
database = 'motd_db'


def test01():
    result = None
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
    # return "성공"


def insert01():

    # data = [ gender, age, result_style ]

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
                        VALUES ("여자", "40대", "섹시")
                    '''
                    # cursor.execute(sql, data)
                    cursor.execute(sql)
                    connection.commit()
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
    # return "성공"


def insert02(gender, age, result_style):

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
                        INSERT INTO `motd_db`.`main_test2` (`gender`, `age`, `result_style`) 
                        VALUES (%s, %s, %s)
                    '''
                    cursor.execute(sql, data)
                    connection.commit()
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
    # return "성공"














# -------------------------
if __name__ == '__main__':
    # select_login('guest', '1')
    print('이 파일을 실행함')
else:
    print('다른 모듈이 가져다가 사용시 호출')
    pass
