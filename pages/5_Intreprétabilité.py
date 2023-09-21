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
    page_title="Interprétabilité",
    #page_icon="👋",
)

#Titre
st.title("Interprétabilité Modèle Benchmark Images Brutes")


st.set_option('deprecation.showPyplotGlobalUse', False)
df=pd.read_csv('Dataframe\df_cleaned.csv')

# Diviser le dataframe en ensembles d'entraînement, de validation et de test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Création du générateur de données et chargement des données
datagen_train = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
datagen_test = ImageDataGenerator(rescale = 1./255)

batch_size = 32
height  = 224 # taille adaptée à VGG16
width = 224
color = 3

train_set = datagen_train.flow_from_dataframe(
    dataframe = train_df,
    directory=None,
    x_col='Path',
    y_col='target',
    target_size=(height, width),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    subset = "training"
)

validation_set = datagen_train.flow_from_dataframe(
    dataframe = train_df,
    directory=None,
    x_col='Path',
    y_col='target',
    target_size=(height, width),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    subset = "validation"
)

test_set = datagen_test.flow_from_dataframe(
    dataframe = test_df,
    directory=None,
    x_col='Path',
    y_col='target',
    target_size=(height, width),
    color_mode = 'rgb',
    classes = None,   # utilise y_col
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = False)

nb_img_train = train_set.samples
nb_img_val = validation_set.samples
nb_img_test = test_set.samples

label_map = train_set.class_indices

#Chargement du modèle Benchmark images brutes
model_benchmark = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_benchmark\model_benchmark_images_brutes.h5')

#création d'un dataframe pour l'interprétabilité
mapping_dict = {value: key for key, value in label_map.items()}
df_interpretation = test_df.copy()

#Prediction
predictions_benchmark = model_benchmark.predict(test_set)
y_pred_benchmark = tf.argmax(predictions_benchmark, axis = 1)

#conversion
arr = np.array(y_pred_benchmark)
converted_arr = np.vectorize(mapping_dict.get)(arr)
df_interpretation['prediction'] = converted_arr
df_interpretation.head(10)

#sélection d'une image

df_new = df_interpretation.loc[df_interpretation['target'] == df_interpretation['prediction']] #je loc sur target = image
lenght =len(df_new)

num = st.slider('Sélectionnez une image à traiter', 0, lenght, 100)
st.write(f"vous avez sélectionné une image de la catégorie:", df_new['target'].iloc[num])

image_path = df_new.iloc[num].Path
img = tf.keras.utils.load_img(image_path, target_size=(height, width))
numpy_image = np.array(img)
numpy_image = numpy_image / 255

from lime.lime_image import LimeImageExplainer

# @st.cache_data
def get_explanation(numpy_image, model, num):
    explainer = LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        image=numpy_image,
        classifier_fn=model.predict,
        top_labels=1,
        num_samples=1000
    )
    dict_explainer = {df_new['target'].iloc[num]: explanation}
    return dict_explainer

# Utiliser la fonction
dict_explainer = get_explanation(numpy_image, model_benchmark, num)

def plot_explainer(dict_explainer):
  from skimage.segmentation import mark_boundaries
  for cle, objet in dict_explainer.items():
    print("Analyse LIME pour une image de la catégorie {}".format(cle))
    temp_1, mask_1 = objet.get_image_and_mask(objet.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    temp_2, mask_2 = objet.get_image_and_mask(objet.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(mark_boundaries(temp_1, mask_1))
    plt.title('SuperPixel pour cette catégorie')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(mark_boundaries(temp_2, mask_2))
    plt.title('Répartition des pixels positifs (vert) et négatifs (rouge)')
    plt.axis('off')
    st.pyplot()

plot_explainer(dict_explainer)

