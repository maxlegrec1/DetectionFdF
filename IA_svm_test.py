from main import execute

if __name__ == "__main__":  
    path_of_the_file = "dataset_fire_esc50/1-4211-A-12.wav"
    model_choice = "SVM"
    print(execute(path_of_the_file, model_choice))