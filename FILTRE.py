
import soundfile
import torchaudio
import scipy
import scipy.signal
'''
def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

TO_FILTER,samplerate=open("noisyfire.wav")
TO_FILTER=TO_FILTER.numpy()[0,:] 

FILTERED=scipy.signal.wiener(TO_FILTER)
soundfile.write('res.wav',FILTERED, samplerate)
'''
def filtre(TO_FILTER):
    return scipy.signal.wiener(TO_FILTER)
