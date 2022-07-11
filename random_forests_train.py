from math import log
from tree import build_tree
import numpy as np
import random as rd
import pickle
'''
用于构建具有多棵树的随机森林。在随机森林算法中随机选择的特征个数通常为k=log2（n），其中n为原数据集中特征的个数
'''
def random_forests_training(data_train,trees_nums):
    trees_results=[]
    trees_features=[]
    n=np.shape(data_train)[1]
    if n>2:
        k=int(log(n-1,2))+1
    else:
        k=1
    #构建每颗树
    for i in range(trees_nums):
        print(f'____{i}____')
        data_samples,feature=chooes_samples(data_train,k)
        tree=build_tree(data_samples)
        trees_results.append(tree)
        trees_features.append(feature)
    return trees_results,trees_features

def chooes_samples(data,k):
    '''从样本中随机选择样本及其特征
    :param data: 原始数据集
    :param k: 选择特征个数
    :return: 被选择的样本和特征
    '''
    m,n=np.shape(data)
    feature=[]
    for j in range(k):
        feature.append(rd.randint(0,n-2))
    index=[]
    for i in range(m):
        index.append(rd.randint(0,m-1))
    data_samples=[]
    for i in range(m):
        data_temp=[]
        for fea in feature:
            data_temp.append(data[index[i]][fea])
        data_temp.append(data[index[i]][-1])
        data_samples.append(data_temp)
    return data_samples,feature


def split_data(data_train,feature):
    '''选择特征
    :param data_train: 训练数据集
    :param feature: 要选择的特征
    :return: 选择出的数据集
    '''
    m=np.shape(data_train)[0]
    data=[]
    for i in range(m):
        data_x_tmp=[]
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data

def cal_correct_rate(data_train,final_predect):
    '''预测模型的正确性'''
    m=len(final_predect)
    corr=0.0
    for i in range(m):
        if data_train[i][-1]*final_predect[i]>0:
            corr+=1
    return corr/m

def save_move(trees_result,trees_feature,result_file,feature_file):
    '''
    保存最终模型
    :param trees_result: 训练好的模型
    :param trees_feature: 每个决策树选择的特征
    :param result_file: 模型保存文件
    :param feature_file: 特征保存文件
    :return:
    '''
    np.savetxt(feature_file, trees_feature, fmt='%i', delimiter=',')

    with open(result_file,'wb') as f:
        pickle.dump(trees_result,f)



def load_data():
    from dbfread import DBF
    data = DBF(r'D:\地图\训练区\西固区样本点.dbf', encoding='GBK', load=True)  # 样本提值到点的shp文件
    a = data.records

    x = []
    y = []
    for i in a:
        grid = i['GRID_CODE']
        b = [i['b1_xigu_'], i['b2_xigu_'], i['b3_xigu_'], i['b4_xigu_'], i['b5_xigu_'], i['b6_xigu_'], i['b7_xigu_'],i['GRID_CODE']]
        x.append(b)
        y.append(grid)
    return x


if __name__=='__main__':
    data_train=load_data()
    data_train=data_train
    print("----导入数据----")
    trees_result,trees_feature=random_forests_training(data_train,20)
    print(trees_feature)
    save_move(trees_result,trees_feature,'result_file.pickle','feature_file.dat')
    print('____模型保存成功____')