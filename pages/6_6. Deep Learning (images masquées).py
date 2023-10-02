import streamlit as st
import numpy as np
import os
import glob
import cv2
from keras import callbacks
import pandas as pd
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.applications import VGG16
from keras.layers import Multiply, Flatten, Dense, Input
from keras.models import Model
from keras.layers import BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import PIL.Image
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Application du modèle",
    #page_icon="👋",
)

#Titre
st.header("Modèles entraînés sur les images masquées")

df=pd.read_csv('Dataframe\df_fusion.csv')

tab1, tab2=st.tabs(["**CNN Benchmark**","**VGG16**"])
with tab1:
    col1,col2 =st.columns([0.55,0.45])
    with col1:
        st.markdown('''
            * Structure du modèle
            ''')
        #Chargement du modèle
        model_benchmark_images_masquees = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_benchmark_images_masquees\model_benchmark_images_masquees.h5')

        ## Présentation du Modèle
        # from io import StringIO
        # import sys

        # # Création d'un objet StringIO pour capturer la sortie de la fonction summary
        # buffer = StringIO()
        # sys.stdout = buffer
        # model_benchmark_images_masquees.summary()
        # sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

        # summary_str = buffer.getvalue()
        # Afficher dans Streamlit
        # st.text(summary_str)
        st.image('streamlit_media\Benchmark_Summary.png')
        

    with col2:
        st.markdown('''
            * Nature des images
            ''')
        indexes = np.random.choice(len(df))
        image_path = df["fusionpath"][indexes]

        # Charger et afficher l'image
        plt.figure(figsize=(5,5))
        image = Image.open(image_path)
        plt.imshow(image)
        
        plt.axis("off")
        st.pyplot()
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * CNN Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Matrice_confusion_benchmark_images_masquees.png')
        st.image('streamlit_media\Rapport_Classification_benchmark_images_masquees.png')
    with col2:
        st.markdown('''
            * CNN Benchmark(Images_brutes)
            ''')
        st.image('streamlit_media\Matrice_confusion_benchmark_images_brutes.png')
        st.image('streamlit_media\Rapport_Classification_benchmark_images_brutes.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * CNN Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_masquees).png')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_masquees)_2.png')
        
    with col2:
        st.markdown('''
            * CNN Benchmark(Images_brutes)
            ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_brutes).png')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_brutes)_2.png')
        
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.1,0.8,0.1])
    with col1:
        st.write("")
    with col2:
        st.subheader('''
            :red[Le modèle perd beaucoup en performance]
            ''')
        st.subheader('''
            :red[Vers un modèle pré-entrainé ? VGG16 Resnet...]
            ''')
    with col3:
        st.write("")

with tab2:
    col1,col2 =st.columns([0.55,0.45])
    with col1:
        st.markdown('''
            * Structure du modèle
            ''')
        #Chargement du modèle
        model_VGG16_images_masquees = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_VGG16_images_masquees\model_VGG16_images_masquees.h5')

        ## Présentation du Modèle
        # from io import StringIO
        # import sys

        # # Création d'un objet StringIO pour capturer la sortie de la fonction summary
        # buffer = StringIO()
        # sys.stdout = buffer
        # model_benchmark_images_masquees.summary()
        # sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

        # summary_str = buffer.getvalue()
        # Afficher dans Streamlit
        # st.text(summary_str)
        st.image('streamlit_media\VGG16_Summary.png')
        

    with col2:
        st.markdown('''
            * Nature des images
            ''')
        indexes = np.random.choice(len(df))
        image_path = df["fusionpath"][indexes]

        # Charger et afficher l'image
        plt.figure(figsize=(5,5))
        image = Image.open(image_path)
        plt.imshow(image)
        
        plt.axis("off")
        st.pyplot()

    st.divider()

    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * VGG16(Images_Masquees)
            ''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_masquees.png')
        st.image('streamlit_media\Rapport_Classification_VGG16_images_masquees.png')
    with col2:
        st.markdown('''
            * VGG16(Images_Brutes)
            ''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_brutes.png')
        st.image('streamlit_media\Rapport_Classification_VGG16_images_brutes.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * VGG16(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_masquees).png')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_masquees)_2.png')
        
    with col2:
        st.markdown('''
            * VGG16(Images_Brutes)
            ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes).png')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes)_2.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.1,0.8,0.1])
    with col1:
        st.write("")
    with col2:
        st.subheader('''
            :red[Modèle VGG16 performant et robuste lorsqu'il est appliqué sur des images masquées]
            ''')   
        

# with tab3:
#     col1,col2 =st.columns([0.55,0.45])
#     with col1:
#         st.markdown('''
#             * Structure du modèle
#             ''')
#         #Chargement du modèle
#         model_VGG16_images_brutes = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_VGG16_images_brutes\model_VGG16_images_brutes.h5')

#         ## Présentation du Modèle
#         # from io import StringIO
#         # import sys

#         # # Création d'un objet StringIO pour capturer la sortie de la fonction summary
#         # buffer = StringIO()
#         # sys.stdout = buffer
#         # model_benchmark_images_masquees.summary()
#         # sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

#         # summary_str = buffer.getvalue()
#         # Afficher dans Streamlit
#         # st.text(summary_str)
#         st.image('streamlit_media\VGG16_Summary.png')
        

#     with col2:
#         st.markdown('''
#             * Nature des images
#             ''')
#         indexes = np.random.choice(len(df))
#         image_path = df["filepath"][indexes]

#         # Charger et afficher l'image
#         plt.figure(figsize=(5,5))
#         image = Image.open(image_path)
#         plt.imshow(image)
        
#         plt.axis("off")
#         st.pyplot()
#     st.divider()
#     col1,col2,col3=st.columns([0.4,0.2,0.4])
#     with col1:
#         st.write("")
#     with col2:
#         st.image('streamlit_media/FlecheBas.png')
#     with col3:
#         st.write("")

#     col1,col2 =st.columns([0.50,0.50])
#     with col1:
#         st.markdown('''
#             * VGG16(Images_brutes)
#             ''')
#         st.image('streamlit_media\Matrice_confusion_VGG16_images_brutes.png')
#         st.image('streamlit_media\Rapport_Classification_VGG16_images_brutes.png')
#     with col2:
#         st.markdown('''
#             * VGG16(Images_Masquees)
#             ''')
#         st.image('streamlit_media\Matrice_confusion_VGG16_images_masquees.png')
#         st.image('streamlit_media\Rapport_Classification_VGG16_images_masquees.png')
    
#     st.divider()
#     col1,col2,col3=st.columns([0.4,0.2,0.4])
#     with col1:
#         st.write("")
#     with col2:
#         st.image('streamlit_media/FlecheBas.png')
#     with col3:
#         st.write("")

#     col1,col2 =st.columns([0.50,0.50])
#     with col1:
#         st.markdown('''
#             * VGG16(Images_brutes)
#             ''')
#         st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes).png')
        
#     with col2:
#         st.markdown('''
#             * VGG16(Images_Masquees)
#             ''')
#         st.image('streamlit_media\Interpretabilite_VGG16(Images_masquees).png')
    
#     st.divider()
#     col1,col2,col3=st.columns([0.4,0.2,0.4])
#     with col1:
#         st.write("")
#     with col2:
#         st.image('streamlit_media/FlecheBas.png')
#     with col3:
#         st.write("")

#     col1,col2,col3=st.columns([0.2,0.6,0.2])
#     with col1:
#         st.write("")
#     with col2:
#         st.markdown('''
#             * Modèle reste très performant
#             ''')   
#         st.markdown('''
#             * Les prédictions restent ciblées sur les zones de l'image \ncorrespondant aux cellules sanguines
#             ''')

