La fonction image_dataset_from_directory() nous aide à charger les données à partir de dossiers organisés de la manière suivante :
    
    train_path/
    ├── happy/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── sad/
    │   ├── img3.jpg
    │   ├── img4.jpg

Elle retourne les images sous forme de tenseurs, groupées en tuples de la forme :

    (image_batches, label_batches)

en considérant les noms des sous-dossiers comme noms de classes.

Par défaut, le type des labels est int, mais grâce au paramètre label_mode= [ 'categorical', 'binary', 'int' ], il est possible de modifier leur format :

 - **int** :  les labels sont des entiers (ex. 0, 1, 2, …)

 - **categorical** : les labels sont encodés sous forme de vecteurs one-hot (ex. [0, 0, 1, 0])

 - **binary** : les labels sont représentés par 0 ou 1 (utile pour la classification binaire)
