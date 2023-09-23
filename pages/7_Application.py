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
st.title("Modèle(Format d'images)")

df=pd.read_csv('Dataframe\df_fusion.csv')

tab1, tab2, tab3, tab4=st.tabs(["**Benchmark(Images masquées)**","**VGG16(Images masquées)**","**VGG16(Images brutes)**","**Synthèse**"])
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
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Matrice_confusion_benchmark_images_masquees.png')
        st.image('streamlit_media\Rapport_Classification_benchmark_images_masquees.png')
    with col2:
        st.markdown('''
            * Benchmark(Images_brutes)
            ''')
        st.image('streamlit_media\Matrice_confusion_benchmark_images_brutes.png')
        st.image('streamlit_media\Rapport_Classification_benchmark_images_brutes.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_masquees).png')
        
    with col2:
        st.markdown('''
            * Benchmark(Images_brutes)
            ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_brutes).png')
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.2,0.6,0.2])
    with col1:
        st.write("")
    with col2:
        st.markdown('''
            * Le modèle perd beaucoup en performance
            ''')
        st.markdown('''
            * Vers un modèle pré-entrainé ? VGG16 Resnet...
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
        st.image('streamlit_media/FlecheBas.png')
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
            * Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Matrice_confusion_benchmark_images_masquees.png')
        st.image('streamlit_media\Rapport_Classification_benchmark_images_masquees.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * VGG16(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_masquees).png')
        
    with col2:
        st.markdown('''
            * Benchmark(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_masquees).png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.2,0.6,0.2])
    with col1:
        st.write("")
    with col2:
        st.markdown('''
            * Modèle beaucoup plus performant
            ''')   
        st.markdown('''
            * Le masquage des images présente-t-il un intérêt ?
            ''')

with tab3:
    col1,col2 =st.columns([0.55,0.45])
    with col1:
        st.markdown('''
            * Structure du modèle
            ''')
        #Chargement du modèle
        model_VGG16_images_brutes = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_VGG16_images_brutes\model_VGG16_images_brutes.h5')

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
        image_path = df["filepath"][indexes]

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
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * VGG16(Images_brutes)
            ''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_brutes.png')
        st.image('streamlit_media\Rapport_Classification_VGG16_images_brutes.png')
    with col2:
        st.markdown('''
            * VGG16(Images_Masquees)
            ''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_masquees.png')
        st.image('streamlit_media\Rapport_Classification_VGG16_images_masquees.png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2 =st.columns([0.50,0.50])
    with col1:
        st.markdown('''
            * VGG16(Images_brutes)
            ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes).png')
        
    with col2:
        st.markdown('''
            * VGG16(Images_Masquees)
            ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_masquees).png')
    
    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.2,0.6,0.2])
    with col1:
        st.write("")
    with col2:
        st.markdown('''
            * Modèle reste très performant
            ''')   
        st.markdown('''
            * Les prédictions restent ciblées sur les zones de l'image \ncorrespondant aux cellules sanguines
            ''')
with tab4:
    import pandas as pd
    import plotly.graph_objects as go

    st.markdown('''
            * Dataframe
            ''')

    data = {
        'Modèle': ['Benchmark(Images_Egal_hist)', 'Benchmark(Images_Egal_adapt)', 'Benchmark(Images_Masquees)', 'Benchmark(Images_brutes)','VGG16(Images_Masquees)','VGG16(Images_brutes)'],
        'Accuracy': [0.9023, 0.9163, 0.9321, 0.9512,0.9830,0.9915],
        'Recall':[0.8704, 0.9179, 0.9281, 0.9451,0.9804,0.9913],
        'Precision':[0.9131, 0.9060, 0.9224, 0.9487,0.9845,0.9911],
        'F1 score':[0.8820, 0.9078, 0.9239, 0.9467,0.9823,0.9912],
    }

    df = pd.DataFrame(data)
    
    st.dataframe(df)

    st.markdown('''
            * Représentation graphique
            ''')
    # Création du graphique avec plotly
    
    fig = go.Figure()

    # Ajout des données pour chaque métrique
    fig.add_trace(go.Bar(x=df['Modèle'], y=df['Accuracy'], name='Accuracy', marker_color='blue'))
    fig.add_trace(go.Bar(x=df['Modèle'], y=df['Recall'], name='Recall', marker_color='green'))
    fig.add_trace(go.Bar(x=df['Modèle'], y=df['Precision'], name='Precision', marker_color='yellow'))
    fig.add_trace(go.Bar(x=df['Modèle'], y=df['F1 score'], name='F1 score', marker_color='red'))

    # Modification de l'échelle du graphique (par exemple, entre 0.8 et 1.0)
    fig.update_yaxes(range=[0.8, 1.0])

    # Ajout du titre et des légendes
    fig.update_layout(title='Performance des modèles', barmode='group')

    st.plotly_chart(fig)
