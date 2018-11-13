from __future__ import division
import numpy as np
from scipy.io import wavfile
import scipy.fftpack


def  xdcthaar(X,L,N):
    for i in range(N-L,N):
        X[i]=0
    return X

def samples():
    rate,array=wavfile.read("0.1_1.wav")
    result=array[0:rate]
    print(result.shape)
    return result

def DCT(x):
    gtm=scipy.fftpack.dct(x,norm='ortho')
    return gtm


def comp(x,L,file_name):
    N=x.shape[0]
    M=N-L
    X=DCT(x)
    print(X)
    print(X.shape)
    f=open(file_name,'w+')
    f.write("%d\n" %N)
    f.write("%d\n" %M)
    for i in range(M):
        f.write("%f\n" %X[i])
    f.close()
    Xm=xdcthaar(X,L,N)
    xm=scipy.fftpack.idct(Xm,norm="ortho")
    print(xm)
        
        
def decomp(file_name):
    f=open(file_name,"r")
    contents=f.readlines()
    N=int(contents[0].strip('\n'))
    M=int(contents[1].strip('\n'))
    contents=contents[2:]
    array=[]
    for i in contents:
        array=np.append(array,float(i.strip('\n')))
    for i in range(M,N,1):
        array=np.append(array,0)

    #print(array.shape)    
    xm=scipy.fftpack.idct(array,norm="ortho")
    print(xm) 




#driver program
L=200
file_name="input.txt"
x=samples()
xup=[]
for i in range(len(x)):
    xup = np.append(xup,x[i][0])
    
comp(xup,L,file_name)
decomp(file_name)       