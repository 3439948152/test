#M=1.2M(sun)
#m=0.5M(木星）,T=4.5
#开普勒方程 二体问题运动方程的一个积分。它反映天体在其轨道上的位置与时间t的函数关系。对于椭圆轨道，开普勒方程可以表示为E－esinE=M，式中E为偏近点角，M为平近点角，（单位都是弧度）都是从椭圆轨道的近地点开始起算，沿逆时针方向为正，E和M都是确定天体在椭圆轨道上的运动和位置的基本量。

from math import*
import numpy as np
import matplotlib.pyplot as plt
M_sun=1.9891*10**30
m_jupiter=1.8982*10**27
M_1=1.2*M_sun
m_1=0.5*m_jupiter
T=6*24*60*60
e=0.6
G=6.67259*10**-11
a=pow((G*M_1*T**2)/4*pi**2,3)
p=a*(1-e**2)#半通径
AU=149597870700



#开普勒方程
M=np.linspace(0,2*pi,8)
n=0
data_M=[]
for j in range(len(M)):
    e = 0.9  # np.linspace(0.5,0.9,5)
    E0 = 0
    E1 = 1

    for i in range(10000):
        if abs(E1-E0)>1e-6:
            a=E1
            E1=M[j]+e*sin(E0)
            E0=a
        else:

            print(E1, i)
            break
    data_M.append(E1)
#print(data_M)

def velocity(m1,m2,a,e,i,f,w):#f为真近点角
    a1 = 28.4392 / sqrt(1 - e ** 2)
    b = m2 * sin(i) / m_jupiter
    c = 1 / sqrt((m1 + m2) / M_sun)
    d = 1 / sqrt(a / 1 * AU)
    g=cos(w+f)+e*cos(w)
    return  a1*b*c*d*g
f=np.linspace(0,2*pi,100)
w=np.linspace(0,pi,8)

for i in w:
    data = []
    for j in f:
        df=velocity(M_sun,m_jupiter,5.0*AU,0.0489,pi/4,i,j)
        data.append(df)
    #plt.plot(f,data)
#plt.show()
#print(df)
def t(w,f,e):

    g = cos(w + f) + e * cos(w)
    return g
df=[]
w=np.linspace(0,2*pi,100)
f=pi/3
for i in range(len(w)):
    a=t(w[i],f,e)
    df.append(a)
plt.plot(w,df)
plt.show()