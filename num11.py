import re
import os
import string
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


output_path=r'D:\space_data\train\data_2'

def readpath(path):
    data=[]

    for dirpath, dirnames, filenames in os.walk(path):
        suf=os.path.splitext(filenames[0])[0]
        suffix = re.split('_', suf)
        print(filenames)
        for filename in filenames:
            suffix = os.path.splitext(filename)[0]
            #suffix=re.split('_',suffix)
            #print(suffix)



            dat_path = os.path.join(dirpath, filename)
            data.append(dat_path)

    return data
#readpath(output_path)


def readpath_2(path):
    data = []
    for dirpath, dirnames, filenames in os.walk(path):
        df = len(filenames)

        for i in range(df):
            file_name = os.path.join(path,f'train_{int(i)}.dat')
            data.append(file_name)

    return data
#旋转90度
def unit_rotate_1(block_unit):
    list_len = len(block_unit)

    # 复制二维数组
    tmp_unit = [[block_unit[j][i] for i in range(list_len)] for j in range(list_len)]

    # 直接逐个坐标点替换数据
    for i in range(list_len):
        for j in range(list_len):
            block_unit[i][j] = tmp_unit[list_len - j - 1][i]
def read_dat(path):
    f = open(path, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.strip()  # .split('\t')
        s = s.split(',')
        sentimentlist.append(s)
    f.close()
    df = [[int(i) for i in l] for l in sentimentlist]
    df=np.array(df)
    return df



path=r'D:\space_data\test_label.dat'
path_2=r'D:\space_data\train\data'

def label_array(path):
    data = []
    df = open(path, encoding='utf-8')
    for i in df:
        s = i.strip().split('\t')
        data.append(s[1])
    df.close()
    data = [int(i) for i in data]
    data = data

    y = np.array(data)
    return y



#print(data[703][1])
#数据旋转，增加数据
def revavl(num):
    input_path=r'D:\space_data\train\data'
    paths=readpath_2(input_path)
    #print(paths)
    for i in range(len(paths)):
        df_arr=read_dat(paths[i])
        unit_rotate_1(df_arr)#旋转90度
        unit_rotate_1(df_arr)  # 再旋转90度
        unit_rotate_1(df_arr)
        num_1=num+i
##四个参数依次为文件名、数组、数据类型（浮点型）、分隔符（逗号）
        filename=f'train_{num_1}.dat'
        #a=[filename,data_2[i][1]]
        #train_label_2.append(a)
        #df = [[int(i) for i in l] for l in train_label_2]

        np.savetxt(os.path.join(output_path, filename), df_arr, fmt = '%i', delimiter = ',')
        print(f'----{filename}保存成功----')
    #np.savetxt('train_label_1.dat',df,fmt = '%s', delimiter = ',')
    #print(f'----train_label保存成功----')

#编写label文件



def prepare(data):

    M = []
    # print(len(data))
    for path in data:
        f = open(path, encoding='utf-8')
        sentimentlist = []
        for line in f:
            s = line.strip()  # .split('\t')
            s = s.split(',')
            sentimentlist.append(s)
        f.close()
        df = [[int(i) for i in l] for l in sentimentlist]
        a = np.array(df)
        img_train = a.tolist()
        M.extend(img_train)
    return M


def lebal_list(path):
    data = []
    df = open(path, encoding='utf-8')
    for i in df:
        s = i.strip().split('\t')
        data.append(s[1])
    df.close()
    data = [int(i) for i in data]
    data = data

    y = np.array(data)
    return y


def label_array(path,path_2):
    df = open(path, encoding='utf-8')
    data = []
    for i in df:
        s = i.strip().split('\t')
        # s = s.split(',')
        data.append(s)
    df.close()
    # print(data)
    data_1 = []
    for dirpath, dirnames, filenames in os.walk(path_2):
        # print(filenames)
        for filename in filenames:
            # suffix = os.path.splitext(filename)[0]
            # print(filename)
            data_1.append(filename)
    data_2 = []
    for i in range(len(data)):
        for j in range(len(data_1)):
            if data[i][0] == data_1[j]:
                data_2.append(data[i][1])
                break
            else:
                continue

    y = np.array(data_2)
    y = [int(i) for i in y]

    return y
def pltshow(path):
    f = open(path, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.strip()
        s = s.split(',')
        sentimentlist.append(s)
    f.close()
    df = [[int(i) for i in l] for l in sentimentlist]
    plt.matshow(df)
    plt.show()
def unit(data):
    data_1=[]

    m,n=np.shape(data)

    for i in range(n):
        data_2=[]
        for j in range(m):
            data_2.append(data[j][i])
            d=np.argmax(np.bincount(data_2))
            #df=np.bincount(data_2)
        data_1.append(d)


    return data_1
