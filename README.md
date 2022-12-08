# DetectionFdF
Ce github contient l'intégralité de notre code sur la detection de feux de forêt.

# Programmes

## main.py
En entrée, choix du modèle et du fichier audio à classifier.
La sortie correspond à la classe du fichier audio (Fire, Not a fire).

## database.py
Permet de pré-formater les données (même durée, même fréquence d'échantillonnage) et de les enregistrer pour pouvoir les utiliser dans les autres programmes

## FILTRE.py
Applique le filtre de Wiener aux signaux sonores

## spectrogramme.py
Permet de réaliser le spectrogramme d'un enregistrement sonore

## resize_db.py
Diminue la taille du spectrogramme pour avoir des données d'entrée pour les modèles moins volumineuses, et sans les axes

## reseau_neurone.py
Réseau de neurones convolutif de classification d'images binaires, entraîné sur les feux, et sur les non-feux

## dataset_for_nn.py
Permet de charger les spectrogrammes pour pouvoir créer les datasets de train, de validation et de test pour les modèles

## conf_matrix.py
Programme permettant de réaliser une matrice de confusion permettant de visualiser les performances d'un modèle, avec le nombre de faux positifs et de faux négatifs

## loading_model.py
Programme permettant de charger un modèle pré-saved pour évaluer ses performances sur des données test