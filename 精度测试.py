# -*- coding:utf-8 -*-
import pandas as pd
from random_forests_test import OA,KP
from sklearn.metrics import confusion_matrix
from dbfread import DBF
#软件
def main():
    data = DBF(r'D:\地图\训练区\西固区测试点_最大似然算法分类.dbf', encoding='GBK', load=True)  # 样本提值到点的shp文件
    a = data.records

    x = []
    y = []
    for i in a:
        grid = i['GRID_CODE']
        b = i['RASTERVALU']
        b = b #+ 1
        # b=abs(b-256)
        x.append(b)
        y.append(grid)

    conf = confusion_matrix(x, y)#x为样本集，y为测试集
    df=pd.DataFrame(conf)
    df.columns=['water','plant ','snow ','building','unused land']
    df.index=['water','plant ','snow ','building','unused land']
    print(df)
    print('混淆矩阵', conf)
    OA_1 = OA(conf)
    print('总体精度：', OA_1)
    kp = KP(OA_1, conf)
    print("KP:", kp)
    file = open(r'C:\Users\Administrator\Desktop\西固区\分类评估\西固区最大似然算法分类精度评估.txt', 'w')
    file.writelines('Overall Accuracy ='+str(OA_1)+'\n')
    file.writelines('Kappa Coefficient =' + str(kp) + '\n')
    file.write('混淆矩阵：\n')
    file.write(str(df))
    file.close()
#算法
def main_2():
    data = DBF(r'D:\地图\训练区\西固区测试点_随机森林算法分类.dbf', encoding='GBK', load=True)  # 样本提值到点的shp文件
    a = data.records

    x = []
    y = []
    for i in a:
        grid = i['GRID_CODE']
        b = i['RASTERVALU']

        b=abs(b-256)
        x.append(b)
        y.append(grid)

    conf = confusion_matrix(x, y)

    df = pd.DataFrame(conf)
    df.columns = ['water', 'plant ', 'snow ', 'building', 'unused land']
    df.index = ['water', 'plant ', 'snow ', 'building', 'unused land']
    print('混淆矩阵：')
    print( df)
    OA_1 = OA(conf)
    print('Overall Accuracy =', OA_1)
    kp = KP(OA_1, conf)
    print("Kappa Coefficient=", kp)
    file = open(r'C:\Users\Administrator\Desktop\西固区\分类评估\西固区随机森林算法分类精度评估.txt', 'w')
    file.writelines('Overall Accuracy =' + str(OA_1) + '\n')
    file.writelines('Kappa Coefficient =' + str(kp) + '\n')
    file.write('混淆矩阵：\n')
    file.write(str(df))
    file.close()
if __name__=='__main__':
    main()
    #main_2()
