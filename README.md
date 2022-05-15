Application pour la sécurité des portes

Auteur : Amaury BONNEAU

Date : mai 2022

# 1) Utilisation de l'application :

## Environnement Python :
-Lancer un invité de commande.

-Se placer dans le dossier de la solution avec la commande `cd [l'adresse du dossier]`

-Créer un environnement python par exemple via la commande `conda create mon_environnement`

-Activer l'environnement avec `conda activate mon_environnement`.

-Lancer l'installation des modules requis avec avec pip : `pip install -r requirement.txt`.

Il est possible d'avoir des difficultés à installer correctement tensorflow-gpu qui est nécessaire à la solution de classification.
Utiliser une installation conda pour ce module si nécessaire (conda prend en charge la récupération des bonnes versions de Cuda/CuDNN).


## Lancement de l'application graphique Streamlit :

Pour lancer localement la solution, se placer à la racine du github et lancer l'application Streamlit avec la commande:
`streamlit run app.py`
Se connecter ensuite via navigateur sur l'URL indiquée dans le command prompt.

# 2) Structure des fichiers :

La solution consiste en trois scripts:

    -app.py : application principale Streamlit
    -detection.py : contient les fonctions qui font la détection et la classification des portes sur image et sur vidéo.
    -utilitaires.py : contient des fonctions utilitaires (prétraitement des images + conversion du format des classes prédites)

Le dossier modele_classif contient le modèle CNN de classification des portes au format protobuf.

Le dossier modele_detection contient les poids du modèle de détection Yolov5

Info : detection_classification.ipynb est un notebook jupyter commenté où l'on peut voir comment le modèle de classification a été créé.
