import cv2
import numpy as np
from utilitaires import *

Label = [' Porte ']
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 180, size=(2, 3))
boxes = []
classid = 0
roi_cropped = None 
TAILLE_CIBLE = (100,100)

def detec_et_classif_image(modele_detection, modele_classif, image, seuil_confiance):
    """Prend en paramètres un modèle de détection, une image et un seuil de confiance.
    Retourne une région d'intérêt de détection de porte (None si pas de détection)."""
    roi_cropped = None 
    boxes = [] 

    # Prédiction par le modèle de détection sur l'image uploadée :
    results = modele_detection(image)

    for i in range(0,len(results.pred[0])) :
        if results.pred[0][i,4] > seuil_confiance :
            
            x = int(results.pred[0][i,0])
            y = int(results.pred[0][i,1])
            w = int(results.pred[0][i,2])
            h = int(results.pred[0][i,3])
            box = np.array([x, y, w, h])
            boxes.append(box)
        
    for box in boxes:
        color = colors[int(classid) % len(colors)]
        
        cv2.rectangle(image, (box[0],box[1]), (box[1]+box[2],box[1]+box[3]), color, 2)
        
        cv2.rectangle(image, (box[0], box[1]), (box[1] + box[2], box[1]+20), color, -1)
        
        # récupération de la portion où une porte est détectée par le modèle
        roi_cropped=image[box[1]:box[3],box[0]:box[2]]
        roi_cropped = pretraiter(roi_cropped, TAILLE_CIBLE)
        classif_pred = modele_classif(roi_cropped)
        texte_pred = convertir_classe(classif_pred)
        
        cv2.putText(image, texte_pred, (box[0], box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))

        roi_cropped=image[box[1]:box[3],box[0]:box[2]] # récupération de la portion où une porte est détectée par le modèle
    
    return roi_cropped, image

def detec_et_classif_video(modele_detection, modele_classif, capture_video, seuil_confiance, largeur_bande:int=100, offset_texte:int=65, taille_texte:float=2.5, epaisseur_texte:int=8):
    """Prend en paramètres un modèle de détection, une capture vidéo et un seuil de confiance.
    Retourne une région d'intérêt de détection de porte (None si pas de détection) et une frame vidéo avec ROI imprimée si détection."""
    roi_cropped = None
    ret, frame = capture_video.read()
    
    boxes = []

    # Correction de l'inversion de couleurs due à cv2 :
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Prédiction par le modèle de détection sur la vidéo uploadée :
    results = modele_detection(frame)

    for i in range(0,len(results.pred[0])) :
        if results.pred[0][i,4] > seuil_confiance :
            
            x = int(results.pred[0][i,0])
            y = int(results.pred[0][i,1])
            w = int(results.pred[0][i,2])
            h = int(results.pred[0][i,3])
            box = np.array([x, y, w, h])
            boxes.append(box)
        
    for box in boxes:
        color = colors[int(classid) % len(colors)]
        
        cv2.rectangle(frame, (box[0],box[1]), (box[1]+box[2],box[1]+box[3]), color, 2)
        
        cv2.rectangle(frame, (box[0], box[1]), (box[1] + box[2], box[1]+largeur_bande), color, -1)
        
        # récupération de la portion où une porte est détectée par le modèle
        roi_cropped=frame[box[1]:box[3],box[0]:box[2]]
        roi_cropped = pretraiter(roi_cropped, TAILLE_CIBLE)
        classif_pred = modele_classif(roi_cropped)
        texte_pred = convertir_classe(classif_pred)

        cv2.putText(frame, texte_pred, (box[0], box[1] + offset_texte), cv2.FONT_HERSHEY_SIMPLEX, taille_texte, (255,255,255), epaisseur_texte)
        #TODO : mettre à l'échelle automatiquement la taille/épaisseur du texte et de la bande de couleur en fonction de la résolution de la webcam/vidéo



    return roi_cropped, frame