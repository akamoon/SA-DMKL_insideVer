from numpy import *
import sys
import numpy.linalg as LA
import numpy as np
import scipy as sp
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn import datasets
from sklearn.decomposition import PCA
import sklearn
import random
from sklearn import svm
from decimal import *
import cv2
class SVM():
    def __init__(self, l , n ):
        self.l= l
        self.n =n
# 定义核函数
    def loadData(fileName):
            dataSet = []
            labels = []
            fr = open(fileName)
            for line in fr.readlines():
                lineArr = line.strip().split(' ')
                n = len(lineArr)
                #print(lineArr[0][0])
                for i in range(n):
                    if lineArr[i][0]=='A':
                        lineArr[i]=lineArr[i][:0]+'0'+lineArr[i][1:]
                #         a[:1] + '1' + a[2:]
                #    elif lineArr[i]=='M':
                #        lineArr[i]='-1'
                fltLine = list(map(float, lineArr[0:n - 1]))
                print(fltLine)
                dataSet.append(fltLine)
                labels.append(float(lineArr[-1]))
            return dataSet, labels
        #定义核函数选择类型


    def exxp(z):
        return (z + np.log(np.exp(-z) + 1)) if z > 0 else np.log(1 + np.exp(z))

    def kernelTrans(X, A, kTup):
            m, n = X.shape
            K = mat(zeros((m, 1)))
            #kTup=''.join(kTup).strip('\n')
            #print(kTup)
            # 线性核
            #这里注意，np矩阵相乘的
            if kTup[0] == 'lin':
                #K = np.matmul(X,A.T)
                K=X*A.T
            #RBF 核函数
            elif kTup[0] == "poly":
                K = pow(np.array(float(kTup[1])*(X * A.T)+float(kTup[2])),float(kTup[2]))
            elif kTup[0] == 'rbf':
                for j in range(m):
                    deltaRow = X[j, :] - A
                    K[j] = deltaRow * (deltaRow.T)
                    #print(float(kTup[1]))
                    #print(round(float(kTup[1])))
                #mark :这里要注意的是‘1.3’如何转化为1.3，只是进行简单强制转换不行，需要加round()才行
                    #print(K[j])
                    df = pd.DataFrame(np.array(SVM.exxp(K[j]/ round(float(-2* float(kTup[1])*float(kTup[1])),6))),columns=list('a'))
                    K[j] = df['a'].apply(Decimal)
            elif kTup[0] == 'lap':
                for j in range(m):
                    deltaRow = X[j, :] - A
                    K[j] = deltaRow * (deltaRow.T)
                    #print(float(kTup[1]))
                    #print(round(float(kTup[1])))
                #mark :这里要注意的是‘1.3’如何转化为1.3，只是进行简单强制转换不行，需要加round()才行
                    #print(K[j])
                    df = pd.DataFrame(np.array(SVM.exxp(K[j]/ round(float(-1* float(kTup[1])),6))),columns=list('a'))
                    K[j] = df['a'].apply(Decimal)
            # Arc-cos 反三角核函数
            #Debug
            #a=np.array([0,3,4,2,6,4]);
            #b = a.reshape((2, 3))
            # print(b)
            # print(np.linalg.norm(b,axis=1,keepdims=True))
            #[[5.        ]
            #[7.48331477]]
            elif kTup[0] == 'tan':
                K = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
            elif kTup[0] == 'Arcos':
                if kTup[2] == 'L0':
                    Theta =np.arccos((np.matmul(X,A.T)) /(LA.norm(X,1,axis=1,keepdims=True)*LA.norm(A,1,axis=1)))
                elif kTup[2] == 'L1':
                    Theta = np.arccos((np.matmul(X,A.T)) /(LA.norm(X,2,axis=1,keepdims=True)*LA.norm(A,2,axis=1)))
                    #Debug
                    #print(LA.norm(X,2,axis=1,keepdims=True)*LA.norm(A,2,axis=1))
                    #print(np.matmul(X,A.T).shape)
                    #print((LA.norm(X,2,axis=1,keepdims=True)*LA.norm(A,2,axis=1)))
                    #print(np.matmul(X,A.T)/(LA.norm(X,2,axis=1,keepdims=True)*LA.norm(A,2,axis=1)))
                    #print(LA.norm(X, 2, axis=0))
                    #print(A)
                    #print(LA.norm(A,2,axis=1))
                    #print(LA.norm(X,2,axis=1,keepdims=True))
                    #print(Theta)
                elif kTup[2] == 'L2':
                    Theta = np.arccos((np.matmul(X,A.T)) /(LA.norm(X, np.inf,axis=1,keepdims=True)*LA.norm(A, np.inf,axis=1)))
                J = SVM.J_solution(Theta,kTup)
                if kTup[2] == 'L0':
                    K = (1 / np.pi) * LA.norm(X, 1,axis=1,keepdims=True)  * LA.norm(A, 1,axis=1)  * J
                elif kTup[2] == 'L1':
                    #Debug
                    #print(1/np.pi)
                    #print((LA.norm(X, 2,axis=1,keepdims=True)  * LA.norm(A,2,axis=1)).shape)
                    K = (1 / np.pi) * LA.norm(X, 2,axis=1,keepdims=True)  * LA.norm(A,2,axis=1)  * J
                elif kTup[2] == 'L2':
                    K = (1 / np.pi) * LA.norm(X, np.inf,axis=1,keepdims=True)  * LA.norm(A, np.inf,axis=1)  * J
            else:
                raise NameError('Unrecognizable Kernel')
            #print(K.shape)
            return K
    def J_solution(Theta,kTup):
        if round(int(kTup[3])) == 0:
            J = np.pi - np.array(Theta)
        elif round(int(kTup[3])) == 1:
            # Debug
            # print(np.sin(Theta))
            # print(np.cos(Theta))
            # mark : 2个一维矩阵相乘需要将中一个矩阵转置才能实现，否则报错
            # mark :
            idx = Theta is nan
            Theta[idx] = 0
            # print(np.cos(np.array(Theta)))
            # print((np.pi-Theta))
            # print((np.cos(Theta)*(np.pi-Theta)))
            J = np.sin(np.array(Theta)) + np.cos(np.array(Theta)) * (np.pi - np.array(Theta))
            # print(J.shape)
            # print(J)
        elif round(int(kTup[3])) == 2:
            # 这个地方我自己球求导出来的结果为
            # J = 2*math.sin(Theta)*math.cos(Theta)+(math.pi-Theta)*(1+2*math.cos(Theta)^2)
            # 看看到时候结果对比哪一个更好，在进行选择
            J = 3 * np.sin(np.array(Theta)) * np.cos(np.array(Theta)) + (np.pi - np.array(Theta)) * (
            1 + 2 * np.cos(np.array(Theta)) ^ 2)
            # print(J)
        return J

    def iteration(X,A,n,l,kTup):
        m, n = X.shape
        iter = 0
        Kl_ii=mat(zeros((m, 1)))
        Kl_ij=SVM.kernelTrans(X,A,kTup)
        #print(Kl_ij.shape, "........")
        for i in range(m):
            if(kTup[0]=='Arcos'):
                Kl_ii[i] = SVM.kernelTrans(X[i,:], X[i, :], kTup)
        Kl_jj = SVM.kernelTrans(A, A,kTup)
        #print(Kl_ii.shape, "........")
        #print(Kl_ii.shape)
        #print(Kl_jj.shape)
        while(iter< round(int(l))):
            Kl_ij = SVM.Deep_kernel(Kl_ij, Kl_ii, Kl_jj, kTup)
            Kl_ii = SVM.Deep_kernel(Kl_ii, Kl_ii, Kl_ii, kTup)
            Kl_jj = SVM.Deep_kernel(Kl_jj, Kl_jj, Kl_jj, kTup)
            #print(Kl_ij)
            #print(Kl_ii)
            #print(Kl_jj)
            iter=iter+1

        return Kl_ij

    def Deep_kernel(Kl_ij,Kl_ii,Kl_jj,kTup):
        Theta = np.arccos(np.array(Kl_ij)*(1/np.sqrt(np.array(Kl_ii)*np.array(Kl_jj))))
        Kl_next_ij=1/np.pi*(1/np.sqrt(np.array(Kl_ii)*np.array(Kl_jj)))*SVM.J_solution(Theta,kTup)
        idx = Kl_next_ij is nan
        Kl_next_ij[idx] = 0
        return Kl_next_ij
        #计算误差
    def calcEk(oS, k):
            fXk = multiply(oS.alphas, oS.Y).T * oS.K[:, k] + oS.b
            return fXk - oS.Y[k]
        #更新误差
    def updateEk(oS, k):
            Ek = SVM.calcEk(oS, k)
            oS.eCache[k] = [1, Ek]
        #第一次选择阿尔法值
    def selectJrand(i, m):
            j = i
            while j == i:
                j = int(random.uniform(0, m))
            return j
        #选择了第一个阿尔法值之后选择第二个阿尔法值
    def selectJ(oS, i, Ei):
            maxK = -1
            maxDelta = -1
            Ej = 0
            oS.eCache[i] = [1, Ei]
            validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
            if len(validEcacheList) > 1:
                for k in validEcacheList:
                    if k == i: continue
                    Ek = SVM.calcEk(oS, k)
                    deltaE = abs(Ek - Ei)
                    if deltaE > maxDelta:
                        maxDelta = deltaE
                        maxK = k
                        Ej = Ek
                        #  search the max ui which is the max distance of svm u=wx+b;
                return maxK, Ej
            else:
                j = SVM.selectJrand(i, oS.m)
                Ej = SVM.calcEk(oS, j)
                return j, Ej

        #第二次选择第二个阿尔法值
    def secondChoiceJ(oS, i):
            nonBounds = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            m = len(nonBounds)
            st = int(random.uniform(0, m))
            for i in range(m):
                j = nonBounds[(st + i) % m]
                if (j == i): continue
                kIi = oS.K[i, i]
                kIj = oS.K[i, j]
                kJj = oS.K[j, j]
                eta = kIi + kJj - 2 * kIj
                if (eta > 0):
                    return j
            bounds = nonzero((oS.alphas.A == 0) + (oS.alphas.A == oS.C))[0]
            m = len(bounds)
            st = int(random.uniform(0, m))
            for i in range(m):
                j = bounds[(st + i) % m]
                if (j == i): continue
                kIi = oS.K[i, i]
                kIj = oS.K[i, j]
                kJj = oS.K[j, j]
                eta = kIi + kJj - 2 * kIj
                if (eta > 0):
                    return j
            return -1
        #内部循环也就是是否违反KKT条件的样本阿尔法值
    def innerL(oS, i):
            Ei = SVM.calcEk(oS, i)
            if ((oS.alphas[i] < oS.C) and (oS.Y[i] * Ei < -oS.eps)) or ((oS.alphas[i] > 0) and (oS.Y[i] * Ei > oS.eps)):
                j, Ej = SVM.selectJ(oS, i, Ei)
                alphaIold = oS.alphas[i].copy()
                alphaJold = oS.alphas[j].copy()
                s = oS.Y[i] * oS.Y[j]
                if s < 0:
                    L = max(0, alphaJold - alphaIold)
                    H = min(oS.C, oS.C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - oS.C)
                    H = min(oS.C, alphaJold + alphaIold)
                if (L == H):
                    return 0
                # eta
                kIi = oS.K[i, i]
                kIj = oS.K[i, j]
                kJj = oS.K[j, j]
                eta = kIi + kJj - 2 * kIj
                if (eta <= 0):
                    return 0
                    j = secondChoiceJ(oS, i)
                    if j < 0: return 0
                    alphaJold = oS.alphas[j].copy()
                    s = oS.Y[i] * oS.Y[j]
                    if s < 0:
                        L = max(0, alphaJold - alphaIold)
                        H = min(oS.C, oS.C + alphaJold - alphaIold)
                    else:
                        L = max(0, alphaJold + alphaIold - oS.C)
                        H = min(oS.C, alphaJold + alphaIold)
                    if (L == H):
                        return 0
                    kIj = oS.K[i, j]
                    kJj = oS.K[j, j]
                    eta = kIi + kJj - 2 * kIj
                aJ = alphaJold + oS.Y[j] * (Ei - Ej) / eta
                if aJ > H:
                    aJ = H
                elif aJ < L:
                    aJ = L
                oS.alphas[j] = aJ

                if (abs(oS.alphas[j] - alphaJold) < oS.eps):
                    return 0
                oS.alphas[i] = alphaIold + s * (alphaJold - oS.alphas[j])
                bi = -Ei - oS.Y[i] * kIi * (oS.alphas[i] - alphaIold) - oS.Y[j] * kIj * (
                oS.alphas[j] - alphaJold) + oS.b
                bj = -Ej - oS.Y[i] * kIj * (oS.alphas[i] - alphaIold) - oS.Y[j] * kJj * (
                oS.alphas[j] - alphaJold) + oS.b
                if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                    oS.b = bi
                elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                    oS.b = bj
                else:
                    oS.b = (bi + bj) / 2.0
                SVM.updateEk(oS, i)
                SVM.updateEk(oS, j)
                return 1
            return 0
        #smo总实现oS为数据集，maxIter等于最大迭代次数，Ktup为核函数参数还有范式的设置
    def smo(oS, maxIter, kTup=('lin', 0 , 1, 1)):
            iter = 0
            entireSet = True
            numChanged = 0
            while (iter < maxIter) and ((numChanged > 0) or (entireSet)):
                numChanged == 0
                # print(numChanged)
                iter += 1
                # print(iter)
                if entireSet:
                    for i in range(oS.m):
                        numChanged += SVM.innerL(oS, i)
                else:
                    nonBounds = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
                    for i in nonBounds:
                        numChanged += SVM.innerL(oS, i)
                if entireSet:
                    entireSet = False
                elif numChanged == 0:
                    entireSet = True

class optStruct:
    def __init__(self, dataSet, labels, C, eps, kTup, l):
        self.X = dataSet
        self.θ = 1
        self.Y = labels.T
        self.C = C
        self.accurate=1.0
        self.RCC=1
        self.eps = eps
        self.m, self.n = shape(dataSet)
        self.alphas = mat(zeros((self.m, 1)))
        #对于α值的初始化对于结果也有很大的影响
        #for i in range(self.m):
        #    if self.Y[i]== -1:
        #        self.alphas[i]=1
        #        break
        for i in range(self.m):
            if self.Y[i]== 1:
                self.alphas[i]=1
                break
        self.betas = mat(zeros((self.m, 1)))
        for i in range(self.m):
            self.betas[i]=float(1/self.m)
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            if(kTup[0]=='Arcos'):
                self.K[:, i] = SVM.iteration(self.X, self.X[i, :],kTup[3],l,kTup)
            else:
                self.K[:, i] = SVM.kernelTrans(self.X, self.X[i, :], kTup)
            #print(self.K[:,i])
def C_M(oS):
    # Cm是f(x)的求导，也就是(wx+b)'= w,Mm就是f(x),也就是wx+b中最小的哪一个。他们都需要满足一个条件就是x<k*sqrt(C)
    # RMC= 4*(sqrt(Cm(48*e*k^2*ln(m+1)))+sqrt(Mm(ln(2/theta)/n)))
    # 其中 Cm=w=sum(α*y*x),这里不方便计算，于是将y=(wx+b)^2,那么 Cm= 2(wx+b)w
    #这里需要注意的是这里的C和M都是常数不是一个函数
    min = 9999999.0000000
    fk = []
    fk_label = []
    fp = []
    k=0
    m=0
    for j in range(oS.m):
        fp.append(abs(multiply(oS.alphas, oS.Y).T * oS.K[:, j] + oS.b))
        x_1 =  abs(oS.K[j, j] * (math.sqrt(oS.C)))
        if fp[j]<x_1:
            fi= abs(1 - oS.Y[j] *fp[j])
            fk.append(fp[j])
            fk_label.append(j)
            m=m+1
            if fi<min and fi!=0:
                min=fi
                k = oS.K[j, j]
    Mm= min
    min=999999999999.00000
    for p in range(oS.m):
        for i in range(m):
            fi = abs(abs(1 - oS.Y[p] * fp[p])-abs(1 - oS.Y[fk_label[i]] * fk[i]))
            fx= abs(fp[p]-fk[i])
            C=float(fi/fx)
            if C<min and C!=0:
                min=C
    Cm=min
    return Cm,Mm,k


def Rademacher_chaos_complex(oS,value):
#SVM 的损失函数可以看作是 L2-norm 和 Hinge loss 之和
#C(t)就是对原函数进行求导
#C=1/λ,λ=1/C,λ为正则化参数，当C大时，表示λ小，则会high variance,low bias(overfitting)
    Cm,Mm,k= C_M(oS)
    print(Cm,Mm,k)
    if value=='rbf' or value =='lap':
        x = math.sqrt((384 * math.e +2)* (k * k) * 5 / (oS.m * float(1 / oS.C)))
    else :
        x = math.sqrt((50 * math.e * (k * k) * math.log(6)) / (oS.m * float(1 / oS.C)))
    c = oS.accurate
    y = math.sqrt(math.log(float(2 / c)) / oS.m)
    RCC=4*(Cm*x+Mm*y)
    return RCC


def T_theta(oS):
    #μ的定义 μ =(K(x1,x1), ...,K(xl,xl))T
    miu = mat(zeros((oS.m, 1)))
    for i in range(oS.m):
        miu[i]=oS.K[i,i]
    Miu=miu.T
    # ν =( Kθ(x1,x1),...,Kθ(xl,xl))T.
    v = mat(zeros((oS.m, 1)))
    for i in range(oS.m):
        v[i]=oS.K[i,i]*oS.θ
    ν=v.T
    #Z = νT* β0 − β0T * Kθ * β0,
    #print((ν*oS.betas).shape)
    Z=ν*oS.betas - oS.betas.T *oS.K *oS.betas
    # −1T*α0(μTβ0 −β0TKkβ0)+Z(α0TKkα0)/2,
    minus= mat(ones((oS.m, 1)))
    minus=minus*-1
    #print((Miu*oS.betas).shape)
    oS.θ=(minus.T*oS.alphas)*(Miu*oS.betas-oS.betas.T*oS.K*oS.betas)+ Z*(oS.alphas.T*oS.K*oS.alphas)


def testRbf(value,value1,value2,value3,value4,data_train,label_train,data_test,label_test,dataSet,labels):
    #kernel='rbf',k1=1.3,L='L1',n=1
    #print(value,value1,value2,value3)
    print ("-----------"+value+"---------")
    oS = optStruct(data_train, label_train, 200, 0.00000001, (value,value1,value2,value3),value4)
    SVM.smo(oS, 100, (value,value1,value2,value3))
    w=0
    for i in range(oS.m):
       w=w+oS.alphas[i]*oS.Y[i]
    #print(w)
    #print(oS.K)
    svInd = nonzero(oS.alphas.A > 0)[0]
    sVs = oS.X[svInd]
    labelSV = oS.Y[svInd]
    print ('There are %d support vectors' % len(svInd))
    errorCount = 0
    predict_T = []
    pred_train_T=[]
    predict1_T = []
    for i in range(oS.m):
        kernelEval = SVM.kernelTrans(sVs, oS.X[i, :], (value,value1,value2,value3))
        predict = kernelEval.T * multiply(labelSV, oS.alphas[svInd]) + oS.b
        pred_train_T.append(predict)
        #print(predict)
        if sign(predict) != sign(oS.Y[i]):
            errorCount += 1
    print ('Training error rate is: %f' % (float(errorCount)/oS.m))
    #dataArr, labelArr = SVM.loadData('testSetRBF2.txt')
    #print(test)
    dataArr = data_test
    labelArr = label_test
    dataArr1 = dataSet
    labelArr1 = labels
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs, dataMat[i, :], (value,value1,value2,value3))
        predict = kernelEval.T * multiply(labelSV, oS.alphas[svInd]) + oS.b
        predict_T.append(predict)
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print ('test error rate is %f' % (float(errorCount)/m))
    oS.accurate=1-(float(errorCount)/m)
    print(oS.accurate)
    RCC=Rademacher_chaos_complex(oS,value)
    oS.RCC=RCC
    print(RCC)
    #ok

    dataMat = mat(dataArr1)
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs, dataMat[i, :], (value,value1,value2,value3))
        predict1 = kernelEval.T * multiply(labelSV, oS.alphas[svInd]) + oS.b
        predict1_T.append(predict1)

    predict2_A= np.array(pred_train_T)
    predict2_T = predict2_A.T
    predict2_C=mat(predict2_T)
    predict2_C=predict2_C.T

    predict_A= np.array(predict_T)
    predict_T = predict_A.T
    predict_C=mat(predict_T)
    predict_C=predict_C.T

    predict1_A= np.array(predict1_T)
    predict1_T = predict1_A.T
    predict1_C=mat(predict1_T)
    predict1_C=predict1_C.T
    #print(predict_C)
    return oS,predict_C,predict1_C,predict2_C
    #beta(oS, 100, (value,value1,value2,value3))
    #print(oS.betas)
    #w=0
    #for i in range(oS.m):
    #   w=w+oS.betas[i]

   # T_theta(oS)
    #print(oS.θ)
    #for i in range(m):
    #    kernelEval = SVM.kernelTrans(sVs, dataMat[i, :], (value, value1, value2, value3))
    #    predict = (abs(oS.θ)*kernelEval.T) * multiply(labelSV, oS.alphas[svInd]) + oS.b
     #   if sign(predict) != sign(labelMat[i]):
     #       errorCount += 1
    #print('test error rate is %f' % (float(errorCount) / m))
    #print(w)
    #print(oS.betas)




def beta_test(value,value1,value2,value3,value4):
    dataSet, labels = SVM.loadData('germen.txt')
    dataSet = mat(dataSet)
    labels = mat(labels)
    oS = optStruct(dataSet, labels, 200, 0.00000000000000001, (value,value1,value2,value3),value4)

    beta(oS, 700, (value,value1,value2,value3))
    #print(oS.betas)
    w=0
    for i in range(oS.m):
       w=w+oS.betas[i]
    print(w)
    #print(oS.betas)

def beta_inner(oS, i):
    if (oS.betas[i] < 1) or ((oS.betas[i] < oS.eps) ):
        j = beta_selectJ(oS, i ,oS.betas[i])
        betaIold = oS.betas[i].copy()
        betaJold = oS.betas[j].copy()
            #L = max(0, alphaJold + alphaIold - oS.C)
        # eta
        #L = max(0, 1-betaJold - betaIold)
        kIi = oS.K[i, i]
        kIj = oS.K[i, j]
        kJj = oS.K[j, j]
        eta = -kIi + kJj - 2 * kIj
        w=0.000
        for k in range(oS.m):
            if k==i :continue
            if k==j :continue
            else:
                w=w+float(oS.betas[k]*(oS.K[2,k]-oS.K[1,k]))

        #print(eta)
        aJ = ((kJj)-w-(betaIold+betaJold)*(kIi+kIj))/(2*eta)
        oS.betas[j] = aJ
        if oS.betas[i]< 0:
            oS.betas[i]=oS.eps
        p=1
        for k in range(oS.m):
            if k==i :continue
            if k==j :continue
            else:
                p=p-oS.betas[k]
        oS.betas[i]=betaIold+betaJold-aJ
        return 1
    return 0


def beta(oS, maxIter, kTup=('lin', 0, 1, 1)):
    iter = 0
    entireSet = True
    numChanged = 0
    while (iter < maxIter) and ((numChanged > 0) or (entireSet)):
        numChanged == 0
        # print(numChanged)
        iter += 1
        # print(iter)
        if entireSet:
            for i in range(oS.m):
                numChanged += beta_inner(oS, i)
        else:
            nonBounds = nonzero((oS.betas.A < 0) )[0]
            for i in nonBounds:
                numChanged += beta_inner(oS, i)
        if entireSet:
            entireSet = False
        elif numChanged == 0:
            entireSet = True
   # print(oS.betas)


def beta_selectJ(oS, i, beta):
    maxK = -1
    maxDelta = -1
    oS.eCache[i] = [1, beta]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            betaJ=oS.betas[k]
            if k == i: continue
            deltaE = abs(betaJ + beta)
            if deltaE > maxDelta:
                maxDelta = deltaE
                maxK = k
                #  search the max ui which is the max distance of svm u=wx+b;
        return maxK
    else:
        j = SVM.selectJrand(i, oS.m)
        return j



def read_data(n):
    if n==0:
        pca = PCA(n_components=2)
        #cancers = datasets.load_breast_cancer()


        #picture
        dataSet2 = []
        labels1= []
        for i in range(637):
            print(str(i+1)+'_'+'-1'+'.jpg')
            img1 = cv2.imread('-1'+' '+ '('+str(i + 1)+')' + '.JPEG')
            img1 = cv2.resize(img1,(64,64),interpolation=cv2.INTER_CUBIC)
            size= img1.shape
            p = []
            for m in range(size[0]):
                for n in range(size[1]):
                    c = max(img1[m, n][0], img1[m, n][1], img1[m, n][2])
                    p.append(c)
            p = np.array(p)
            dataSet2.append(p)
            labels1.append(-1)
            if (i <= 251):
                img2 = cv2.imread('1'+' '+ '('+str(i + 1)+')' + '.JPEG')
                print(str(i + 1) + '_' + '1' + '.jpg')
                img2 = cv2.resize(img2, (64, 64),interpolation=cv2.INTER_CUBIC)
                size1 =img2.shape
                k = 0
                q = []
                for m in range(size1[0]):
                    for n in range(size1[1]):
                        c = max(img2[m, n][0], img2[m, n][1], img2[m, n][2])
                        # print(c)
                        q.append(c)
                q = np.array(q)
                dataSet2.append(q)
                labels1.append(1)
        dataSet2 = np.array(dataSet2)




        #dataSet, labels1 = cancers['data'], cancers['target']
        #dataSet2 = pca.fit_transform(dataSet2)
        for i in range(dataSet2.shape[0]):
            if labels1[i]!=1:
                labels1[i]=-1
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataSet2, labels1, random_state=1,train_size=0.6)
        dataSet = mat(x_train)
        labels = mat(y_train)
        dataSet1=mat(x_test)
        labels2 =mat(y_test)
        dataSet2=mat(dataSet2)
        labels3=mat(labels1)
        return dataSet,labels,dataSet1,labels2,dataSet2,labels3


if __name__=="__main__":
    #mark: 这里需要注意，readline()和input 的区别，readline()在读入数据的时候需要注意用line.strip()去掉结尾的\n
    print ("number of Candidate kernel :")
    value5 = input()
    print ("number of model layers:")
    value7 = input()
    value = []
    value1 = []
    value2 = []
    value3 = []
    value4 = []
    for i in range(int(value5)):
        print ("input kerel :")
        value.append(input())
        print ("input k : ")
        value1.append(input())
        print ("input L : ")
        value2.append(input())
        print ("input n : ")
        value3.append(input())
        print ("input Arcos layer :")
        value4.append(input())
    n = 0
    m = 0
    list1 = []
    predict = []
    predict1 = np.array([0])
    whole1 =np.array([0])
    data_train,label_train,data_test,label_test,dataSet,labels=read_data(0)
    while (m < int(value7)):
        n=0
        while(n < int(value5)):
           # dataSet, labels = SVM.loadData('testSetRBF.txt')
           # dataSet = mat(dataSet)
            #labels = mat(labels)
            #oS = optStruct(dataSet, labels, 0, 0, (0,0,0,0),0)
            oS,pre,whole,test= testRbf(value[n],value1[n],value2[n],value3[n],value4[n],data_train,label_train,data_test,label_test,dataSet,labels)

            list1.append(oS)
            if n ==0 and oS.RCC<10000:
                pred =np.array(pre)
                whole2 =np.array(whole)
                test1 =np.array(test)
            elif n!=0 and oS.RCC<10000:
                pred =np.concatenate((pred, pre), axis=1)
                whole2 =np.concatenate((whole2,whole),axis=1)
                test1 =np.concatenate((test1,test),axis=1)
            #print(predict[0])
            n += 1
            #print(list1[n - 1].RCC)
            #beta_test(value, value1, value2, value3, value4)
        m+=1
        data_train = test1 
        dataSet= whole2
        data_test = pred
        print(whole2)
    #print(pred.shape)
    #oS,pre=testRbf(value[0],value1[0],value2[0],value3[0],value4[0],0,pred)
    #pred = np.concatenate((pred, pre), axis=1)
    #oS, pre = testRbf(value[0], value1[0], value2[0], value3[0], value4[0], 0, pred)





