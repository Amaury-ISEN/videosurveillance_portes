import cv2
import numpy as np

def pretraiter(data, cible):
    """Prend une liste d'images ou une image seule en couleurs et retourne la liste d'image ou l'image redimensionnées
    à la taille cible (tuple de int pour width et height) et en nuance de gris."""
    # Cas d'un array ou d'une liste d'images :
    if type(data) == list:
        result = []
        for img in data:
            img = cv2.resize(img,cible)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 2)
            result.append(img)
    # Cas d'une image seule:
    else:
        result = cv2.resize(data,cible)
        result= cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Ajout d'une dimension de 1 en fin d'array de l'image pour le canal de couleur unique (nuances de gris) 
        result = np.expand_dims(result, 2)
        # Ajout d'une dimension au début de l'array de l'image pour fonctionner comme un batch de size 1 dans le predict du classifieur
        result = np.expand_dims(result, 0)

    return result

def convertir_classe(pred):
    """Convertit un array de prédiction en une classe textuelle."""
    if np.argmax(pred) == 0:
            str_pred = 'porte fermee' # TODO: si on veut un accent à "fermée" il faut rebuild cv2 avec un module freetype d'opncv-contrib...
    elif np.argmax(pred) == 1:
        str_pred = 'porte ouverte'
    else:
        str_pred = 'porte entrouverte'

    return str_pred
