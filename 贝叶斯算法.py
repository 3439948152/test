import numpy as np
from math import*
from random_forests_test import readTif,writeTiff,unit
import os
import pandas
def load_data():
    from dbfread import DBF
    data = DBF(r'D:\地图\训练区\西固区样本点.dbf', encoding='GBK', load=True)  # 样本提值到点的shp文件
    a = data.records

    x = []
    y = []
    for i in a:
        grid = [i['b1_xigu_'], i['b2_xigu_'], i['b3_xigu_'], i['b4_xigu_'], i['b5_xigu_'], i['b7_xigu_']]

        b = [i['b1_xigu_'], i['b2_xigu_'], i['b3_xigu_'], i['b4_xigu_'], i['b5_xigu_'], i['b7_xigu_'],i['GRID_CODE']]
        y.append(b)
        x.append(grid)
    return y

def func(b1):
    mean=np.mean(b1)
    ax=0
    for i in b1:
        ac=(i-mean)**2
        ax=ac+ax
    ax=ax/(len(b1)-1)
    std=sqrt(ax)
    return mean,std
def trans(df):
    model=[]
    df = np.swapaxes(df, 0, 1)
    for i in df:
        std, mean = func(i)
        model.append([std, mean])
    return model
def train(data):
    '''
    :param data: 样本数据
    :return:期望与标准差的集合
    '''
    df=[]
    df2=[]
    df3=[]
    df4=[]
    df5=[]
    model=[]
    for i in data:
        if i[6] == 1:
            df.append(i[0:6])
        elif i[6]==2:
            df2.append(i[0:6])
        elif i[6]==3:
            df3.append(i[0:6])
        elif i[6]==4:
            df4.append(i[0:6])
        else:
            df5.append(i[0:6])
    model1=trans(df)
    model2 = trans(df2)
    model3 = trans(df3)
    model4 = trans(df4)
    model5 = trans(df5)
    model.append(model1)
    model.append(model2)
    model.append(model3)
    model.append(model4)
    model.append(model5)

    return model
def predect(model,x):
    fin=[]
    for j in range(len(x)):
        df=[]
        df2=[]
        for q in range(len(model)):
            final = 1
            for i in range(len(model[q])):
                mean,std=model[q][i]
                fx1 = 1 / (std * pow(2 * pi, 0.5)) * np.exp(-((x[j][i] - mean) ** 2) / (2 * std ** 2))
                final=final*fx1
                df2.append(final)
            df.append([final,q+1])
        max_1=max(df2)
        for i in range(len(df)):
            if max_1==df[i][0]:
                a1=df[i][1]
                break
        fin.append(a1)
    #print(len(fin))
    return fin
def predect_1(model,x):
    fin=[]
    for j in range(len(x)):
        df=[]
        df1=[]
        for q in range(len(model)):
            df2 = []
            #final = 1
            for i in range(len(model[q])):

                mean,std=model[q][i]
                fx1 = 1 / (std * pow(2 * pi, 0.5)) * np.exp(-((x[j][i] - mean) ** 2) / (2 * std ** 2))
                #final=final*fx1
                df2.append(fx1)
            final=max(df2)
            df1.append(final)
            df.append([final,q+1])
        max_1 = max(df1)
        for i in range(len(df)):
            if max_1 == df[i][0]:
                a1 = df[i][1]
                break
        fin.append(a1)


    return fin

def main():
    Landset_Path = r"D:\地图\西固区.dat"
    SavePath = r"C:\Users\Administrator\Desktop"
    dataset = readTif(Landset_Path)
    Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
    Tif_height = dataset.RasterYSize  # 栅格矩阵的行数

    Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset.GetProjection()  # 获取投影信息
    Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)


    data = np.zeros((6, Landset_data.shape[1] * Landset_data.shape[2]))
    for i in range(Landset_data.shape[0]-2):
        data[i] = Landset_data[i].flatten()
    data[-1]=Landset_data[-1].flatten()
    data = data.swapaxes(0, 1)

    data1=load_data()
    model = train(data1)
    print('模型训练')
    pred=predect(model,data)
    print('预测')
    pred = np.array(pred)
    pred = pred.reshape(Landset_data.shape[1], Landset_data.shape[2])
    pred = pred.astype(np.uint8)
    #  将结果写到tif图像里
    writeTiff(pred, Tif_geotrans, Tif_proj, os.path.join(SavePath, 'save_1.tif'))
    print("分类结束")

if __name__=='__main__':
    main()












