
from math import*
import numpy as np
import matplotlib.pyplot as plt
M_sun=1.9891*10**30
m_jupiter=1.8982*10**27
#木星径向速度12.7m/s
G=6.67259*10**-11
AU=149597870700

e=[0.4,0.5,0.6,0.7,0.8,0.9]

#a_1=a*m_jupiter/(M_sun+m_jupiter)
m1=1.2*M_sun
m2=0.5*m_jupiter

T=10#周期为四天


#周期与a的关系
def transformT_a(t):

    a1=t**2/365**2#单位为天
    #print (a1)
    a=a1**(1/3)
    return a*AU


#真近点角f与偏近点角E公式
def transf_E(E,e):
    a=tan(E/2)
    b=sqrt((1-e)/(1+e))
    f=2*atan(a/b)
    return f
#开普勒公式(偏近点E与平近点角M)
def kaipule(M,e):
    E0 = 0
    E1 = 1

    for i in range(10000):
        if abs(E1 - E0) > 1e-6:
            a = E1
            E1 = M + e * sin(E0)
            E0 = a
        else:

            #print(E1)
            break
    return E1

def velocity(m1,m2,a,e,i,f,w):#f为真近点角,w为近点幅角

   v1=sqrt(G/((m1+m2)*a*(1-e**2)))*m2*sin(i)
   v2=cos(w+f)+e*cos(w)
   return v1*v2

#print(transformT_a(4)/AU)
#主函数
def main(m1,m2,T,e,i,w,num):
    M = np.linspace(0, 3 * pi, 150)
    t = T * M / (2 * pi)
    df=[]
    for i1 in M:
        a=transformT_a(T)
        E=kaipule(i1,e)
        f=transf_E(E,e)
        v=velocity(m1,m2,a,e,i,f,w)
        df.append(v)
    print(max(df),min(df))
    plt.subplot(3,2,num)
    plt.plot(t,df)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel('时间/天')
    plt.ylabel('radial velocity / m/s')
    #plt.title(f'偏心率{e},公转周期为{T}天的径向速度图像')
    #plt.show()
i=np.linspace(0,pi/2,6)
print(i)
if __name__ == '__main__':
    for n in range(len(i)):
        main(m1,m2,T,0.5,i[n],pi/4,n+1)
    plt.suptitle(f'质量为{m2}kg,公转周期为{T}天的径向速度图像')
    plt.show()


