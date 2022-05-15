import streamlit as st
import tensorflow as tf
import torch
import numpy as np
import cv2
from utilitaires import *
from detection_classification import *
import tempfile # pour la mise en RAM

###################
# INITIALISATIONS #
###################
fichier_photo = None
fichier_video = None
choix_option = None
roi_cropped = None
streamlit_slider_confidence = None

# IMPORT DES MODELES (vérification s'ils sont en clés de session streamlit pour ne pas les refresh à chaque action de l'utilisateur)
#importer le modèle de détection, le modèle est pris sur le git de yolo, les poids pour les portes dans le path local
if 'modele_yolo' not in st.session_state:
    st.session_state['modele_yolo'] = torch.hub.load('ultralytics/yolov5', 'custom', path='modele_detection/best.pt', force_reload=True)
#importer le modèle de classification
if 'modele_classif' not in st.session_state:
    st.session_state['modele_classif'] = tf.keras.models.load_model('.\\modele_classif')

st.set_page_config(page_title="Vidéosurveillance des portes") # texte affiché dans l'onglet
st.title("Vidéosurveillance des portes 🚪", anchor=None) # titre en haut de la page

st.write("""
Cette interface repose sur un pipeline IA qui peut détecter une porte sur une image, une vidéo ou un flux webcam et classer
la porte selon trois classes : ouverte, fermée et entrouverte.
""")

choix_option = st.selectbox(
        'Veuillez choisir une option :',
        ('Prédiction sur photo', 'Prédiction sur vidéo', 'Prédiction sur webcam'))

#########
# PHOTO #
#########

if choix_option == 'Prédiction sur photo':

    st.write("Prédiction sur photo :")
    fichier_photo = st.file_uploader("Veuillez choisir une image.", type='png')
    #les fichiers uploadés sont restitués sous forme de buffers qu'il faut lire pour les utiliser dans nos fonctions

    # Si une photo est uploadée :
    if fichier_photo is not None:
        # On lit le buffer :
        fichier_photo_bytes = fichier_photo.read()

        # Passag de l'image en array numpy :
        fichier_photo_bytes = np.asarray(bytearray(fichier_photo_bytes), dtype=np.uint8)
        img = cv2.imdecode(fichier_photo_bytes, cv2.IMREAD_COLOR) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # correction de l'inversion RGB

        img = np.asarray(img)

        streamlit_slider_confidence = st.slider(label="Seuil de détection :",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.5,
                                                step=0.01)

        # Prédiction par le modèle de détection sur l'image uploadée :
        roi_cropped, img = detec_et_classif_image(modele_detection=st.session_state['modele_yolo'],
                                           modele_classif=st.session_state['modele_classif'],
                                           image=img,
                                           seuil_confiance=streamlit_slider_confidence)
        st.image(img)

#########
# VIDEO #
#########

if choix_option == 'Prédiction sur vidéo':
    st.write("Prédiction sur vidéo :")
    #Widget de téléversement de vidéo pour la classif sur vidéo :
    fichier_video = st.file_uploader("Veuillez choisir une vidéo.", type='mov')

    # Si une vidéo est uploadée :
    if fichier_video is not None:    
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(fichier_video.read())
        cap = cv2.VideoCapture(temp_video.name)
        image_streamlit = st.empty()
        streamlit_slider_confidence = st.slider(label="Seuil de détection :",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.5,
                                                step=0.01)

        streamlit_texte_pred = st.empty()
        while choix_option == 'Prédiction sur vidéo':
            # Détection
            roi_cropped, frame = detec_et_classif_video(modele_detection=st.session_state['modele_yolo'],
                                                 modele_classif=st.session_state['modele_classif'],
                                                 capture_video=cap,
                                                 seuil_confiance=streamlit_slider_confidence)
            # Affichage de l'image avec détection éventuelle :
            image_streamlit.image(frame)

##########
# WEBCAM #
##########

if choix_option == 'Prédiction sur webcam':
    #Widget de téléversement d'image pour la classif sur flux Webcam :
    st.write("Prédiction sur Webcam :")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        streamlit_webcam = st.empty()
        streamlit_slider_confidence = st.slider(label="Seuil de détection :",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.5,
                                                step=0.01)

        image_streamlit = st.empty()
        while choix_option == 'Prédiction sur webcam':
            # Détection
            roi_cropped, frame = detec_et_classif_video(modele_detection=st.session_state['modele_yolo'],
                                                        modele_classif=st.session_state['modele_classif'],
                                                        capture_video=cap,
                                                        seuil_confiance=streamlit_slider_confidence,
                                                        taille_texte=0.5,
                                                        epaisseur_texte=2,
                                                        largeur_bande=20,
                                                        offset_texte=10)

            # Affichage de l'image avec détection éventuelle :
            image_streamlit.image(frame)