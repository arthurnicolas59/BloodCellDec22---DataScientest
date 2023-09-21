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

st.set_page_config(
    page_title="Segmentation",
    page_icon="📊",
)

#Segmentation
st.header("Segmentation")

df_mask=pd.read_csv('Dataframe\df_mask_SAM.csv')
tab1, tab2 = st.tabs(["**Segmentation SAM**", "**Prédiction Unet**"])

with tab1:
    col1,col2=st.columns([0.5,0.5])
    with col1 :
        st.write("**Sélection d'une zone d'intérêt**")
        
    with col2:
        st.write("**Extraction de l'objet**")
    st.image('streamlit_media\SAM2.png')
    st.image('streamlit_media\SAM3.png')
    st.image('streamlit_media\SAM4.png')
    st.image('streamlit_media\SAM5.png')

    import plotly.express as px

    # Triez les catégories target par décompte décroissant
    sorted_targets = df_mask['nameLabel'].value_counts().index.tolist()

    # Utilisez plotly pour créer un histogramme avec les catégories triées
    fig = px.histogram(df_mask, x='nameLabel', title='Histogramme répartition des images par classe', category_orders={"nameLabel": sorted_targets})

    # Affichez le graphique dans Streamlit
    st.plotly_chart(fig)


with tab2:
    st.write('**Prédiction Unet**')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    

    ### Définition d'une fonction de perte : Coefficient de Dice
    def LossDice(y_true, y_pred):
        numerateur  =tf.reduce_sum(y_true*y_pred, axis=(1, 2))
        denominateur=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
        dice=2*numerateur/(denominateur+1E-4)
        return 1-dice

    def DiceCoeff(y_true, y_pred):
        numerateur  =tf.reduce_sum(y_true*y_pred, axis=(1, 2))
        denominateur=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
        dice=2*numerateur/(denominateur+1E-4)
        return dice

    ### Création du jeu d'entrainement, de validation et de test

    height=256
    width=256
    batch_size=32
    seed=42

    #### séparation du jeu de données
    from sklearn.model_selection import train_test_split

    filepaths = df_mask['filepath'].values
    maskpaths = df_mask['maskpath'].values

    train_filepaths, test_filepaths, train_maskpaths, test_maskpaths = train_test_split(filepaths, maskpaths, test_size=0.2, random_state=seed)
    train_filepaths, val_filepaths, train_maskpaths, val_maskpaths = train_test_split(train_filepaths, train_maskpaths, test_size=0.2, random_state=seed)


    nb_img_train=len(train_filepaths)
    nb_img_val=len(val_filepaths)
    nb_img_test=len(test_filepaths)

    ### Chargement éventuel d'un modèle enregistré

    custom_objects = {'DiceCoeff': DiceCoeff, 'LossDice': LossDice}
    saved_segmenter = tf.keras.models.load_model(
        r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_unet_with_data_augmented\model_unet_with_data_augmented.h5', 
        custom_objects=custom_objects
    )

    import tensorflow as tf
    def load_image(filepath,resize=(256,256)):
        # Charger l'information brute en mémoire
        im = tf.io.read_file(filepath)
        # Décoder l'information en un tensorflow RGB (3 channels).
        im = tf.io.decode_jpeg(im, channels=3)
        #Redimensionner l'image
        return tf.image.resize(im, size=resize)

    def load_mask(filepath,resize=(256,256)):
        im = tf.io.read_file(filepath)
        ### Dans le cas des masques, l'image est en noir et blanc, il n'y a donc qu'une valeur par pixel
        im = tf.io.decode_png(im, channels=1)
        return tf.image.resize(im, size=resize)

    ### Affichage d'une prédiction de masque en comparaison avec un masque
    ### fait à partir de l'image segmentée du jeu de donnée Test

    size=2
    indexes=np.random.choice(len(test_filepaths),size=2)

    i=0
    for idx in indexes:
        image=load_image(test_filepaths[idx])
        img=tf.reshape(image,(1,256,256,3))
        mask_pred=saved_segmenter.predict(img)
        mask_true=load_mask(test_maskpaths[idx])
        image=tf.cast(image,dtype=tf.int32)

        plt.figure(figsize=(15,7))

        i+=1
        plt.subplot(size,3,i)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Image d'origine")

        i+=1            
        plt.subplot(size,3,i)
        plt.imshow(tf.reshape(mask_true,(256,256)),cmap='gray')
        plt.axis("off")
        plt.title("Masque issu de SAM")

        i+=1
        plt.subplot(size,3,i)
        plt.imshow(tf.reshape(mask_pred,(256,256)),cmap='gray')
        plt.axis("off")
        plt.title("Masque predit par le modèle Unet")

        st.pyplot()