import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os
from tqdm import tqdm
""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(
        samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i+1])], axis=1)
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
    return newspec, freqs


""" plot spectrogram"""


def plotstft(samples, samplerate, binsize=2**10, plotpath=None, colormap="jet"):
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    #print("timebins: ", timebins)
    #print("freqbins: ", freqbins)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto",
               cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in (
        (xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        #plt.show()
        plt.clf()
    return ims

def plot_with_path(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samples, samplerate = sf.read(audiopath)
    samples=samples[:,0]
    return plotstft(samples, samplerate, binsize=2**10, plotpath=plotpath, colormap="jet")

"""ims = plot_with_path(
"Dataset/Training/Autre13230-0-0-1.wav",plotpath="Dataset_spec/Training/Autre13230-0-0-1.jpg")"""


def create_spec_db(INPUT_DB_PATH,OUTPUT_DB_PATH,beg_index):
    directory = os.fsencode(INPUT_DB_PATH)
    listd_dir=os.listdir(directory)
    listd_dir_sorted=sorted(listd_dir,key=lambda x:os.fsdecode(x))
    #print('Autre13230-0-0-1.wav'>='Autre104998-7-9-9.wav')
    #print(os.fsdecode(listd_dir_sorted[744])>=os.fsdecode(listd_dir_sorted[0]))
    for fichier in tqdm(listd_dir_sorted[beg_index:]):
        fichier_nom=os.fsdecode(fichier)
        #print(fichier_nom)
        plot_with_path(INPUT_DB_PATH+"/"+fichier_nom,plotpath=OUTPUT_DB_PATH+"/"+fichier_nom[:-4]+".jpg")

def get_beginning_ind(OUTPUT_DB_PATH):
    directory = os.fsencode(OUTPUT_DB_PATH)
    index=len(os.listdir(directory))
    return index


if __name__=='__main__':
    INPUT_DB_PATH="Dataset/Test"
    OUTPUT_DB_PATH="Dataset_spec/Test"
    start_index=get_beginning_ind(OUTPUT_DB_PATH)
    create_spec_db(INPUT_DB_PATH,OUTPUT_DB_PATH,start_index)
    print("done")