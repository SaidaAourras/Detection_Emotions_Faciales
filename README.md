# Détection d’Émotions Faciales

## I - Context Géneral

Ce projet consiste à développer un prototype d’API capable de détecter un visage sur une photo, prédire l’émotion correspondante et enregistrer le résultat dans une base de données.
L’objectif est d’évaluer la faisabilité d’une future solution SaaS d’analyse émotionnelle pour des tests produits et des expériences UX.

## II - Planification

| N°  | Tâche                                                | Description courte |
|----|-------------------------------------------------------|--------------------|
| 1  | Documentation des outils et notions                   | Étudier les technologies et concepts nécessaires au projet. |
| 2  | Préparation et exploration des données                | Analyser et nettoyer les données utilisées pour l’entraînement. |
| 3  | Entraînement du CNN                                   | Créer et entraîner le modèle de classification des émotions. |
| 4  | Détection de visages (OpenCV & Haar Cascade)          | Implémenter la détection automatique de visages sur les images. |
| 5  | Création de l’API FastAPI                             | Développer l’API pour recevoir les images et retourner les prédictions. |
| 6  | Tests unitaires & GitHub Actions                      | Mettre en place les tests et l’intégration continue. |

## III - Arborescence de l'Architecture du Projet

        Detection_Emotions_Faciales/
        ├── backend/
        │   ├── CNN_model.keras/
        │   ├── database.py
        │   ├── main.py
        │   └── models.py
        ├── CNN/
        │   ├── images/
        │   ├── Analyse_kagglehub_data.ipynb
        │   ├── detect_and_predict.py
        │   └── haarcascade_frontalface_default.xml
        ├── .gitignore
        ├── README.md
        ├── requirements.txt
        └── test_unitaire.py

## IV - Technologies

- **Backend :**

        - fastapi
        - httpx
        - postresql
        - sqlalchemy
        - python-dotenv
        - pytest
        - joblib

- **Convolutional Neural Network (CNN) :**

        - kagglehub
        - numpy
        - tensorflow / keras
        - matplotlib
        - opencv-python

## V - Installation 
- Cloner le projet :

        git clone https://github.com/SaidaAourras/Detection_Emotions_Faciales.git
        cd Detection_Emotions_Faciales

- Installer les dépendances :

        pip install -r requirements.txt

- Lancer l’API FastAPI :

        uvicorn backend.main:app --reload

- Exécuter les tests unitaires :

        pytest -v

## VII – Utilisation

- Route POST /predict_emotion : envoyer une image pour obtenir l’émotion prédite.

- Route GET /history : consulter l’historique des prédictions dans PostgreSQL.

- Script detect_and_predict.py : détecte le visage et affiche la prédiction sur l’image.

## VII - Fonctionnalités

- Détection de visage automatique

- Prédiction des émotions (happy, sad, angry, surprised, etc.)

- Stockage des résultats dans PostgreSQL

- API REST fonctionnelle

- Tests unitaires et CI/CD avec GitHub Actions
