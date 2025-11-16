# PROJET : Classifieur de textes à partir d’un RNN

Ce projet a pour objectif de concevoir un classifieur d’émotions à partir de phrases Xcourtes, en utilisant un réseau de neurones récurrent (RNN) implémenté avec PyTorch.

<p align="center">
  <img src="data/emotion.png" alt="image" width="500" height="300">
</p>

## mode descirption

- mode 1 : oneword : teste le modèle sur un seul mot (vérifie la lecture d’un token isolé)
- mode 2 : shortseq : teste le modèle sur une courte séquence de 3 mots (vérifie la récurrence).
- mode 3 : dataloader : vérifie le fonctionnement des DataLoaders (batchs, longueurs, padding).
- mode 4 : train : entraîne complètement le modèle sur le jeu d’entraînement et évalue sur le test.

## structure du projet

├── config.py # Paramètres globaux (hyperparamètres, chemins, device, etc.)
├── dataset/ # Données texte : train / validation / test
│ ├── train.txt
│ ├── val.txt
│ └── test.txt
├── feeling/ # Package principal : modèle + dataset + fonctions d’entraînement
│ ├── core.py # Contient RNNManual, OneHotDataset, et les fonctions de training
│ └── init.py
├── scripts/
│ └── run.py # Script principal : exécution selon le mode choisi
├── utils.py # Fonctions utilitaires (graphiques, métriques, statistiques)
├── demo_modes.ipynb # Notebook de démonstration (tests des 4 modes)
├── logs/ # Partie recherche des hyper-paramètres
├── requirements.txt # Liste des dépendances Python

## installation

1. creer l envitonnement virtuel
2. installer les dépendance : pip install -r requirements.txt

## Execution

- soit directement en ligne de commande : python scripts/run.py
- soit par le notebook

note : dans le dépot le dossier dataset a été supprimer pour qu'on puisse le déposer sur tomuss
