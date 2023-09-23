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
st.title("Modèles(Format d'images)")

df=pd.read_csv('Dataframe\df_fusion.csv')

tab1, tab2, tab3, tab4=st.tabs(["**Benchmark(Images masquées)**","**VGG16(Images masquées)**","**VGG16(Images brutes)**","**Synthèse**"])
with tab1:
    col1,col2 =st.columns([0.5,0.5])
    with col1:
        st.markdown('''
            * Structure du modèle
            ''')
        #Chargement du modèle
        model_benchmark_images_masquees = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_benchmark_images_masquees\model_benchmark_images_masquees.h5')

        ## Présentation du Modèle
        from io import StringIO
        import sys

        # Création d'un objet StringIO pour capturer la sortie de la fonction summary
        buffer = StringIO()
        sys.stdout = buffer
        model_benchmark_images_masquees.summary()
        sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

        summary_str = buffer.getvalue()

        # Afficher dans Streamlit
        st.text(summary_str)

    with col2:
        st.markdown('''
            * Sur images fusionnées
            ''')
        indexes = np.random.choice(len(df))
        image_path = df["fusionpath"][indexes]

        # Charger et afficher l'image
        image = Image.open(image_path)
        plt.imshow(image)
        
        plt.axis("off")
        st.pyplot()
