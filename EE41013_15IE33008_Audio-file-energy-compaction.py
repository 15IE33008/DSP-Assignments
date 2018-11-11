import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, dct, idct
from scipy.io import wavfile # get the api
import numpy as np
from IPython.display import Audio
import math

fs, audio = wavfile.read('Sample_audio.wav')
Audio('Sample_audio.wav')

print(f"Audio Type: {audio.dtype}")
print(f"Samples = {audio.shape[0]}   Channels = {audio.shape[1]}") #output=(samples,channels)
print(f"Sampling frequency = {fs} Hz")


audio = audio.T[0]
samples = audio.shape[0]
L = (samples / fs)*1000
print(f'Audio length: {L:.0f} mili-seconds')

f, ax = plt.subplots()
ax.plot((np.arange(samples) / fs)*1000, audio)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Amplitude');

def Error_dft(x,L): #calculating error dft for x
    y_dft = fft(x)
    N = len(y_dft)
    m = int((N+1-L)/2)
    n = int((N-1+L)/2)
    for i in range(m,n+1):
        y_dft[i] = 0
    x_m = ifft(y_dft)

    return ((x - x_m) ** 2).mean(axis=0)

def Error_dct(x,L): #calculating error dct for x
    y_dct = dct(x)
    N = len(y_dct)
    for i in range(N-L,N):
        y_dct[i] = 0
    x_m = idct(y_dct)/(2*len(x))

    return ((x - x_m) ** 2).mean(axis=0)

h2 = np.array([[1,1],[1,-1]])
def haar_mat(n):
    n = int(n)
    if n == 1:
        return h2
    else:
        a = np.kron(haar_mat(n-1),[1,1])
        b = np.kron(np.identity(int(math.pow(2,n-1)))*math.pow(2,(n-1)/2.0),[1,-1])
        #print(np.concatenate((a,b),axis=0))
        return np.concatenate((a,b),axis=0)

def haar(x):
    return np.matmul(haar_mat(math.log(len(x),2)),np.transpose(x))

def ihaar(y): #calculating inverse haar for x
    n = int(math.log(len(y),2))
    N = len(y)
    hn = haar_mat(n)
    return np.matmul(np.transpose(hn)/N,np.transpose(y))

def Error_haar(x,L): #calculating error for haar transform
    y = haar(x)
    N = len(y)
    for i in range(N-L,N):
        y[i] = 0
    x_m = ihaar(y)

    return ((x - x_m) ** 2).mean(axis=0)

n = len(audio)
dft = fft(audio).real

plt.plot(dft)
plt.title('Complete DFT plot')
plt.ylabel('mod values of DFT')
plt.xlabel('samples')
plt.show()

dct_audio = dct(audio)

plt.plot(dct_audio)
plt.title('Complete DCT plot')
plt.ylabel('mod values of DCT')
plt.xlabel('samples')
plt.show()

haar_audio = haar(audio[0:8192])

plt.plot(haar_audio)
plt.title('Complete Haar plot')
plt.ylabel('mod values of Haar')
plt.xlabel('samples')
plt.show()


edft = [0.]*len(audio)
edct = [0.]*len(audio)
ehaar = [0.]*len(audio)

for L in range(len(audio)):
    edft[L] = Error_dft(audio,L)
    edct[L] = Error_dct(audio,L)
    ehaar[L] = Error_dct(audio[0:8192],L)
    
    
    
# Comparision
fig, ax = plt.subplots()
ax.plot(edft, 'k:', label='DFT') 
ax.plot(edct,'k--', label = 'DCT') 
ax.plot(ehaar, 'k', label='HAAR')
ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')