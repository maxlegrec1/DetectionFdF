from re import T
import soundfile
import matplotlib.pyplot as plt
import torchaudio
import torch
import numpy as np
import scipy.fftpack as sf
import scipy
import scipy.signal
def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
data,samplerate=open("Waves.wav")

TO_FILTER,samplerate=open("noisyfire.wav")
TO_FILTER=TO_FILTER.numpy()[0,:] 

FILTERED=scipy.signal.wiener(TO_FILTER)
soundfile.write('res.wav',FILTERED, samplerate)
