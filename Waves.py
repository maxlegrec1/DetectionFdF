import soundfile as sf
import matplotlib.pyplot as plt
data, samplerate = sf.read('waves.wav') 
print(data)
plt.plot(data[:,0])
#plt.plot(data[:,1])
plt.show()
print(samplerate)
#samplerate = 44100
#fréquence d'échantillage de 40 kHz



data[2000][1]= 2

sf.write('new_file.wav', data, samplerate)

