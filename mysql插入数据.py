
# -- coding: utf-8 --
#批量往数据库插入数据
import csv
import pymysql
import time
from pymysql.converters import escape_string
#获取当前时间time1
#time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
#打开数据文档
file=open(r'C:\Users\Administrator\Desktop\景点.csv',encoding='utf-8')
table=csv.reader(file)
#连接数据库
db = pymysql.connect(host = '127.0.0.1' # 连接名称，默认127.0.0.1
,user = 'root' # 用户名
,passwd='cfq14789' # 密码
,port= 3306 # 端口，默认为3306
,db='data_point' # 数据库名称
,charset='utf8' # 字符编码
)
cursor = db.cursor()
print("a")
for row in table:
    #需要执行的sql脚本
    sql = "INSERT INTO travel_travel_point VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    #将插入的数据赋予一个对象
    data=(row[0],row[1],escape_string(row[2]),)
    
    try:
        # 执行 SQL 语句
        cursor.execute(sql,row)
        print('sueecc')
        # 提交修改
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()
        print('failure')
#关闭游标和数据库
cursor.close()
db.close()

