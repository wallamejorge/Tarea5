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


#Modulo 0: Recopilacion de datos y grafica t vs N

files_Data=pylab.loadtxt('monthrg.dat')
tiempo=(files_Data[:,0]*12.0)+files_Data[:,1]
tiempo=tiempo/12.0
media_mensual=files_Data[:,3]
for i in range(len(media_mensual)):
 if media_mensual[i]==(-99):
  media_mensual[i]=0
print "Arreglo media mensual"
plt.plot(tiempo,media_mensual,'ro')
plt.show()
print "grafico t vs N"
#Modulo 1: Transformada discreta de fourier

from scipy.fftpack import fft, fftfreq
print "importo"
fft_x = fft(media_mensual)/128 # FFT Normalized
print "hizo fft"
freq = fftfreq(128,1/1200) # Recuperamos las frecuencias
print "hizo freq"
plt.plot(freq,np.abs(fft_x))
plt.show()

