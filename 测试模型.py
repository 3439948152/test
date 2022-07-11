import numpy as np
import 图像分类
import pickle
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt
#画混淆矩阵
def pltshow(array_1):
    plt.matshow(array_1)
    for i in range(len(array_1)):
        for j in range(len(array_1)):
            plt.annotate(array_1[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.show()
test_path=r'D:\space_data\test'
test_label=r'D:\space_data\test_label.dat'
file = open(r'D:\space_data\random_forest.pickle', "rb")
#把模型从文件中读取出来
model = pickle.load(file)
#关闭文件
file.close()
dict_label={0:'0',1:'1'}
data_2=图像分类.readpath_2(test_path)
pred_img=图像分类.prepare(data_2)
pred_img = np.array(pred_img).reshape(len(data_2), 22500)
predictions=model.predict(pred_img)
pred_labels=图像分类.label_array(test_label,test_path)
#print(pred_labels)
for i in range(88):
    print("第"+str(i)+'张图像预测为'+str(predictions[i]))
df=confusion_matrix(predictions,pred_labels)
print('正确率为：',model.score(pred_img, pred_labels))
print('混淆矩阵',df)