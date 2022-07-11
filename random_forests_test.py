import numpy as np
import os
from tree import predict
from random_forests_train import split_data
from sklearn.metrics import confusion_matrix
import pickle
from osgeo import gdal
def load_data():
    from dbfread import DBF
    data = DBF(r'D:\地图\训练区\西固区测试点.dbf', encoding='GBK', load=True)  # 样本提值到点的shp文件
    a = data.records

    x = []
    y = []
    for i in a:
        grid = i['GRID_CODE']
        b = [i['b1_xigu_'], i['b2_xigu_'], i['b3_xigu_'], i['b4_xigu_'], i['b5_xigu_'], i['b6_xigu_'], i['b7_xigu_']]
        x.append(b)
        y.append(grid)
    return x,y


def load_model(result_file,feature_file):
    '''导入随机森林保存文件数据
    :param result_file:
    :param feature_file:
    :return:
    '''
    f = open(feature_file, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.strip()  # .split('\t')
        s = s.split(',')
        sentimentlist.append(s)
    f.close()
    df = [[int(i) for i in l] for l in sentimentlist]
    df = np.array(df)
    with open(result_file, 'rb') as f:
        trees_result=pickle.load(f)
    return trees_result,df
def get_predict(trees_result,trees_feature,data_train):
    '''利用训练好的随机森林模型对样本进行预测
        :param trees_result: 随机森林模型
        :param trees_feature: 分类树的特征
        :param data_train: 预测样本
        :return: 预测结果
        '''
    m_tree=len(trees_result)
    m=np.shape(data_train)[0]
    result=[]
    for i in range(m_tree):
        clf=trees_result[i]
        feature=trees_feature[i]
        data=split_data(data_train,feature)
        result_i=[]
        for i in range(m):
            result_i.append((list(predict(data[i][0:-1],clf).keys()))[0])
        result.append(result_i)
    return result
#矩阵转置
def unit(data):
    pred = np.array(data)
    df1 = np.swapaxes(pred, 0, 1)

    data_1=[]
    for i in range(len(df1)):
        data_1.append(np.argmax(np.bincount(df1[i])))
    return data_1

#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset

#保存tif文件函数
def writeTiff(im_data,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
#总体精度
def OA(Confusion_Matrix):
    CM = Confusion_Matrix
    xia_list = []
    for i in range(len(CM)):
        xia_list.append(sum(CM[i]))
    xia = sum(xia_list)
    shang_list = []
    for i in range(len(CM)):
        shang_list.append(CM[i][i])
    shang = sum(shang_list)
    OA = shang / xia
    return OA
#Kappa指数
def KP(Overall_Accuracy, Confusion_Matrix):
    CM = Confusion_Matrix
    p0 = Overall_Accuracy
    pe_xia_list = []
    for i in range(len(CM)):
        pe_xia_list.append(sum(CM[i]))
    pe_xia = sum(pe_xia_list) ** 2
    CMT = list(map(list, zip(*CM))) # 转置
    pe_shang_list = []
    for i in range(len(CM)):
        pe_shang_list.append(sum(CM[i]) * sum(CMT[i]))
    pe_shang = sum(pe_shang_list)
    pe = pe_shang / pe_xia
    KP = (p0 - pe) / (1 - pe)
    return KP
'''对遥感影像进行分类并且将分类结果保存为栅格文件，
'''
def main():
    path_1 = r'D:\space_data\遥感影像分类\result_file.pickle'
    path_2 = r'D:\space_data\遥感影像分类\feature_file.dat'
    trees_result, trees_feature = load_model(path_1, path_2)
    Landset_Path = r"D:\地图\西固区.dat"
    SavePath = r"C:\Users\Administrator\Desktop"
    dataset = readTif(Landset_Path)
    Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
    Tif_height = dataset.RasterYSize  # 栅格矩阵的行数
    print(Tif_height, Tif_width)
    Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset.GetProjection()  # 获取投影信息
    Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)
    data = np.zeros((Landset_data.shape[0], Landset_data.shape[1] * Landset_data.shape[2]))
    for i in range(Landset_data.shape[0]):
        data[i] = Landset_data[i].flatten()
    data = data.swapaxes(0, 1)
    pred = get_predict(trees_result, trees_feature, data)
    pred = unit(pred)
    pred = np.array(pred)
    pred = pred.reshape(Landset_data.shape[1], Landset_data.shape[2]) * 255
    pred = pred.astype(np.uint8)
    #  将结果写到tif图像里
    writeTiff(pred, Tif_geotrans, Tif_proj, os.path.join(SavePath, 'save_1.tif'))
    print("分类结束")
#计算分类结果的混淆矩阵，总体精度，Kappa系数
def main_2():
    data_train, label = load_data()
    data = data_train
    label_1 = label
    trees_result, trees_feature = load_model('result_file.pickle', 'feature_file.dat')
    predicton = get_predict(trees_result, trees_feature, data)

    df = unit(predicton)

    conf = confusion_matrix(label_1, df)
    print('混淆矩阵', conf)
    OA_1 = OA(conf)
    print('总体精度：', OA_1)
    kp = KP(OA_1, conf)
    print("KP:", kp)


if __name__=='__main__':
    #main()
    main_2()


