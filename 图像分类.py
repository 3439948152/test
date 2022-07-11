import os


import numpy as np

import matplotlib.pyplot as plt
input_path=r'D:\space_data\data'
output_path=r'D:\空间资料和实验'
test_path=r'D:\space_data\test'
label_path1=r'D:\space_data\train_label.dat'
test_label=r'D:\space_data\test_label.dat'
def readpath(path):
    data=[]
    for dirpath, dirnames, filenames in os.walk(path):
        df = len(filenames)

        for i in range(df):
            file_name = os.path.join(path, f'train_{int(i)}.dat')
            data.append(file_name)

    return data

def readpath_1(path):
    data=[]
    for dirpath, dirnames, filenames in os.walk(path):
        df = len(filenames)

        for i in range(df):
            file_name = os.path.join(path, f'train_{704+int(i)}.dat')
            data.append(file_name)

    return data

def readpath_2(path):
    data=[]
    for dirpath, dirnames, filenames in os.walk(path):
        df = len(filenames)

        for i in range(df):
            file_name = os.path.join(path, f'test_{int(i)}.dat')
            data.append(file_name)

    return data
#print(readpath(input_path))
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


#画混淆矩阵
def pltshow_hunxiao(array_1):
    plt.matshow(array_1)
    for i in range(len(array_1)):
        for j in range(len(array_1)):
            plt.annotate(array_1[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.show()

#标签
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
def lebal_list(path):
    data=[]


    df = open(path, encoding='utf-8')

    for i in df:
        s = i.strip().split('\t')
        data.append(s[1])
    df.close()
    data = [int(i) for i in data]
    data=data+data+data+data
    y=np.array(data)
    return y
train_label_1=r'D:\space_data\train_label.dat'

y=lebal_list(train_label_1)


#图像数据预处理
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
data = readpath(r'D:\space_data\train\data_2')
X1=prepare(data)

X=X1
X = np.array(X).reshape(len(data),22500)#, 150, 150)#+len(data2)
import num11
data_2 = readpath_2(test_path)
pred_img = prepare(data_2)
pred_img = np.array(pred_img).reshape(len(data_2), 22500)
pred_labels = num11.pred_label
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pickle

def main():
    train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y,
                                                                                      random_state=1, train_size=0.7,
                                                                                      test_size=0.3)
    # 用100个树来创建随机森林模型，训练随机森林
    classifier = RandomForestClassifier(n_estimators=150,
                                        bootstrap=True)
    classifier.fit(train_data,  train_label)
    print("训练集：",classifier.score(train_data,  train_label))
    print("准确率：",classifier.score( test_data,test_label))
    file = open(r'D:\space_data\random_forest.pickle', "wb")
    # 将模型写入文件：
    pickle.dump(classifier, file)
    # 最后关闭文件：
    file.close()
    print('随机森林模型训练结束')

def SVM_model():
    from sklearn import svm
    train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y,
                                                                                      random_state=1, train_size=0.8,
                                                                                      test_size=0.2)

    classifier = svm.SVC(C=0.5, kernel='rbf', gamma=2, decision_function_shape='ovo')
    classifier.fit(train_data, train_label)
    print("训练集：", classifier.score(train_data, train_label))
    print("准确率：", classifier.score(test_data, test_label))

    file = open(r'D:\space_data\SVM.pickle', "wb")
    # 将模型写入文件：
    pickle.dump(classifier, file)
    # 最后关闭文件：
    file.close()
    print('训练结束')


import BpNN
def BP():
    train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y,
                                                                                      random_state=1, train_size=0.8,
                                                                                      test_size=0.2)
    bp = BpNN.BP(x_count=784, mid_count=10, y_count=10, eta=0.3, train_count=100, precision=0.001)
    bp.train(train_data, train_label)

    # 训练结束后，用测试数据进行预测
    Y_predict = bp.predict(test_data)
    print(len(Y_predict))





if __name__ == '__main__':
    #main()
    SVM_model()

