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

years=[]
manchas=[]

for i in range(len(media_mensual)):
 if files_Data[i,0]>1800:
  years.append(tiempo[i])
  manchas.append(media_mensual[i])

plt.plot(years,manchas, 'ro')
plt.ylabel('Numero de Machas vs Tiempo')
plt.grid()
plt.show()

#Modulo 1: Transformada discreta de fourier
n=len(manchas)
fft_x = fft(manchas)/n # FFT Normalized
timestep = 1.0/12.0
freq=fftfreq(n, d=timestep)
'''
plt.plot(freq,np.abs(fft_x))
plt.ylabel('Potencias vs Frecuencias')
plt.scatter(freq,np.abs(fft_x))
plt.grid()
plt.show()
'''
#Modulo 2: Espectros de potencias vs f
f_shifted=np.fft.fftshift(freq)
fftx_shifted=np.fft.fftshift(fft_x)
Pot=np.abs(fftx_shifted)*np.abs(fftx_shifted)

plt.plot(f_shifted,Pot)
plt.ylabel('Potencias vs Frecuencias')
plt.scatter(f_shifted,Pot)
plt.grid()
plt.show()

#Modulo 3: Espectros de potencias vs T

T=[]
new_Pot=[]
T0=1
Tf=20
Temp=0

half_n=np.ceil(n/2.0)
fft_x_half=(2.0)*fft_x[:half_n]
freq_half=freq[:half_n]

for i in range(1,len(freq_half)):
    Temp=1.0/freq_half[i]
    if Temp>T0:
        if Temp<Tf:
            T.append(Temp)
            new_Pot.append(np.abs(fft_x_half[i])*np.abs(fft_x_half[i]))
 
plt.plot(T,new_Pot)
plt.ylabel('Potencias vs Tiempo')
plt.scatter(T,new_Pot)
plt.grid()
plt.show()

#Modulo 4: Filtro en Frecuencias
frecuencias=[]
potencias=[]

F0=0.0
Ff=0.15

fft_x[np.abs(freq) > Ff] = 0.0
fft_x[np.abs(freq) < F0] = 0.0

frecuencias=freq
Potencias=np.abs(fft_x)*np.abs(fft_x)

plt.plot(frecuencias,Potencias)
plt.ylabel('Potencias vs frecuencias filtradas')
plt.scatter(frecuencias,Potencias)
plt.grid()
plt.show()

#Modulo 5:Antitransformo
y_real=[]
t=years
y_modelo=[]
clean_f = np.fft.ifft(fft_x)

for i in range(len(manchas)):
    y_real.append(manchas[i])
    y_modelo.append(np.real(clean_f[i])/np.amax(np.real(clean_f)) * np.amax(manchas))

plt.plot(t,y_real,'r',t,y_modelo,'g')
plt.show()

#Modulo 6:Prediccion
#buscar el ultimo maximo

a=0
ini=0
for a in range(len(t)):
    if(t[a]==1995.0):
        ini=a
pos=[]
for i in range(1,100):
    if(y_modelo[ini-i]<y_modelo[ini-i+1]):
        pos.append(ini-i)
  
amax=t[pos[0]]
resultado=amax+10.83
mensaje= "La proxima fecha donde se presentara el mayor numero de manchas solares es:"
anho= int(resultado)
mes= abs(resultado)- anho
#calcular mes
for i in range(12):
    if (i/12.0<=mes):
        if(mes<(i+1)/12.0):
            if(i==0):
                mess= "Enero"
            if(i==1):
                mess= "Febrero"
            if(i==2):
                mess= "Marzo"
            if(i==3):
                mess= "Abril"
            if(i==4):
                mess= "Mayo"
            if(i==5):
                mess= "Junio"
            if(i==6):
                mess= "Julio"
            if(i==7):
                mess= "Agosto"
            if(i==8):
                mess= "Septiembre"
            if(i==9):
                mess= "Octubre"
            if(i==10):
                mess= "Noviembre"
            if(i==11):
                mess= "Diciembre"
print mensaje
print mess
print anho
