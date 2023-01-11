#import soundfile
#import torchaudio
import scipy
import scipy.signal

#filtre le signal en appliquant le filtre de Wiener
def filtre(TO_FILTER):
    return scipy.signal.wiener(TO_FILTER)
