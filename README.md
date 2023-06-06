# OCR - Projet 7 - Implémenter un modèle de scoring
## Description du projet
Le projet consiste à développer un modèle de scoring pour prédire la probabilité de faillite d'un client et à construire un dashboard interactif pour les chargés de relation client. L'objectif est de fournir une explication transparente des décisions d'octroi de crédit et de permettre aux clients d'accéder à leurs informations personnelles.
## Fichiers
### dashboard.py
Ce fichier contient le code pour le tableau de bord interactif développé avec Streamlit. Il récupère les données clients à partir du tableau de bord et les transmet à l'API pour effectuer des prédictions. Il inclut également des graphiques pour la visualisation des données. Le lien vers le dashboard est disponible [ici](https://ocr-db-pret.herokuapp.com).

### api.py
Ce fichier contient le code de l'API développée avec FastAPI. Il utilise un modèle de régression logistique pour réaliser des prédictions à partir des données reçues. Les fichiers api.py et dashboard.py ont été déployés sur Heroku pour permettre les prédictions dans le cloud. Le lien vers l'API est disponible [ici](https://ocr-api-pret.herokuapp.com).

### dashboard_local.py
Ce fichier est similaire à dashboard.py, à la différence qu'il n'interagit pas avec une API. Toute la logique de prédiction est incluse dans le fichier. L'application Streamlit correspondante est disponible à [cette adresse](https://hugodrf88-ocrprojet7-dashboard-local-ojtzf5.streamlit.app)

### data_drift.ipynb  
Ce notebook a été utilisé pour effectuer une analyse de dérive des données entre les données de référence "application_train.csv" et les données de production "application_test.csv" à l'aide de la bibliothèque Evidently. 

### data_drift.html  
Ce fichier est le résultat graphique de l'analyse de dérive des données effectuée dans data_drift.ipynb.

### projet7.ipynb
Ce notebook contient l'ensemble du travail réalisé sur la base de données principale "application_train.csv". Il comprend l'analyse exploratoire des données, l'exploration des corrélations, la recherche d'un modèle de classification binaire adapté, ainsi que l'optimisation des hyperparamètres.

### modular_code.py
Ce script reprend le code précédent, mais sous forme d'un script Python indépendant. Il facilite la localisation des erreurs et comprend des fonctionnalités telles que l'entraînement du modèle sélectionné.

### test_api.py 
Ce fichier contient des tests unitaires réalisés sur api.py à l'aide de pytest.

### Procfile
Ce fichier est nécessaire pour le déploiement du projet sur Heroku.

### requirements.txt
Ce fichier répertorie les packages nécessaires pour exécuter le projet.

## Dossiers
### data
Ce dossier contient les bases de données utiles pour le projet. Certaines d'entre elles sont des échantillons des bases de données d'origine en raison des limitations de taille de fichier sur GitHub.

### functions
Ce dossier contient les fonctions créées spécifiquement pour ce projet, telles que fill_num et fill_cat, ainsi que des métriques personnalisées spécifiques à notre domaine d'application.

### images
Ce dossier contient les graphiques générés lors de l'analyse exploratoire des données à partir du script modular_code.py.

### models
Ce dossier contient les différents modèles d'apprentissage automatique entraînés pour les besoins du projet.

