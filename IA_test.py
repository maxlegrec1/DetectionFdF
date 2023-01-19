import soundfile as sf

def load_file(path):    #chargement du fichier audio
    sig, sr = sf.read(path)
    return (sig, sr)

if __name__ == "__main__":
    path_of_the_file1 = input("Path of the file ")
    (sound1, sr1) = load_file(path_of_the_file1)

    path_of_the_file2 = input("Path of the file ")
    (sound2, sr2) = load_file(path_of_the_file2)

print(sound1[10000,0])
print(sound2[10000,0])