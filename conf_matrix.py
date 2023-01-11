import matplotlib.pyplot as plt
import numpy as np

# Fonction qui affiche la matrice de confusion
def show_confusion_matrix(matrix, labels):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(matrix)
    
    N = len(labels)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    plt.show()
    