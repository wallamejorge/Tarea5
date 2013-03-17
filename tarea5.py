#Tarea No.5

#Importar librerias
import numpy as np
import sys,codecs,math
import os
from numpy import mean,cov,double,cumsum,dot,array,rank
from pylab import plot,subplot,axis,stem,show,figure
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pylab
from scipy.fftpack import fft, fftfreq
#Modulo 0: Recopilacion de datos y grafica t vs N

files_Data=pylab.loadtxt('monthrg.dat')
tiempo=(files_Data[:,0]*12.0)+files_Data[:,1]
tiempo=tiempo/12.0
media_mensual=files_Data[:,3]
for i in range(len(media_mensual)):
 if media_mensual[i]==(-99):
  media_mensual[i]=0
plt.plot(tiempo,media_mensual,'ro')
#plt.show()
#Modulo 1: Transformada discreta de fourier

n=media_mensual.size
fft_x = fft(media_mensual)/n # FFT Normalized
timestep = 0.1
freq=fftfreq(n, d=timestep)
plt.plot(freq,np.abs(fft_x))
#plt.show()
#Modulo 2: Espectros de potencias vs f

f=freq/(2*np.pi)
Pot=np.abs(fft_x)*np.abs(fft_x)
plt.plot(f,Pot)
#plt.show()
#Modulo 3: Espectros de potencias vs T
T=[]
new_Pot=[]
new_f=[]
T0=1
Tf=20
Temp=0
for i in range(1,(len(f)/2)-1):
 Temp=1/f[i]
 if Temp>T0:
  if Temp<Tf:
   T.append(1/f[i])
   new_Pot.append(Pot[i])



