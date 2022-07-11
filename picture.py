import os
import tensorflow as tf
from tensorflow import keras
import keras
from sklearn import model_selection
import numpy as np
from sklearn.metrics import confusion_matrix
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


def lebal_list(path,path_2):
    data=[]
    data_2=[]

    df = open(path, encoding='utf-8')
    df_2 = open(path_2, encoding='utf-8')
    for i in df:
        s = i.strip()   .split('\t')
        data.append(s[1])
    df.close()
    data = [int(i) for i in data]



    for i in df_2:
        s1 = i.strip()#.split('\t')
        s1 = s1.split(',')
        data_2.append(s1[1])
    df_2.close()
    data_2 = [int(i) for i in data_2]
    data=data+data+data+data

    y=np.array(data)
    return y
train_label_1=r'D:\space_data\train_label.dat'
train_label_2=r'D:\space_data\train_label_2.dat'
y=lebal_list(train_label_1,train_label_2)


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

X=X1#+X2
X = np.array(X).reshape(len(data), 150, 150)
train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y,
                                                                                      random_state=1, train_size=0.8,
                                                                                      test_size=0.2)
def creat_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(150, 150)),
        keras.layers.Dense(150, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)

    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data, train_label, epochs=40)
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print('test acc', test_acc)
    # 将模型写入文件：
    model.save('my_model.h5')
    print('训练结束')


#预测




#测试
def pretest():
    new_model = tf.keras.models.load_model('my_model.h5')
    dict_label = {0: '0', 1: '1'}
    data_2 = readpath_2(test_path)
    pred_img = prepare(data_2)
    pred_img = np.array(pred_img).reshape(len(data_2), 150, 150)
    pred_labels=label_array(r'D:\space_data\test_label.dat')
    #pred_labels = label_array(test_label)
    #print(pred_labels)
    jieguo = []
    for i in range(88):
        imgs = pred_img[i]
        img = (np.expand_dims(imgs, 0))

        predictions_single = new_model.predict(img)
        jieguo.append(dict_label[np.argmax(predictions_single)])
        print("第" + str(i) + '张图像预测为' + dict_label[np.argmax(predictions_single)])
    new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    test_loss, test_acc = new_model.evaluate(pred_img, pred_labels)
    print('正确率为：', test_acc)
    pred_labels = [int(l) for l in pred_labels]
    jieguo = [int(i) for i in jieguo]
    pltshow_hunxiao(confusion_matrix(pred_labels, jieguo))
    print('混淆矩阵', confusion_matrix(pred_labels, jieguo))
if __name__ == '__main__':
    creat_model()
    #pretest()
