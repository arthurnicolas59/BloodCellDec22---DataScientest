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
    page_title="Segmentation2",
    page_icon="📊",
)

#EDA
st.header("Segmentation2")


st.set_option('deprecation.showPyplotGlobalUse', False)
df_mask=pd.read_csv('Dataframe\df_mask_Unet.csv')

st.write(df_mask)
### Génération de masques à partir des images segmentées du jeu de données

### Premièrement, on récupère les chemins des dossiers contenant les images



path= df_mask['maskpath']

plt.figure(figsize=(15,7))
plt.subplot(121)



# plt.subplot(122)
example2=cv2.imread(path[20],cv2.IMREAD_GRAYSCALE)
plt.imshow(example2*255,cmap='gray')
plt.title("Masque généré à partir de l'image segmentée")
plt.axis('off');

st.write(example2.min(), example2.max())
st.pyplot()

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

# fonction de chargement pour le modèle de segmentation

def parse_function_segmentation(filepath, maskpath):
    # Lire l'image
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0 # Normaliser entre 0 et 1

    # Lire le masque
    mask = tf.io.read_file(maskpath)
    mask = tf.image.decode_png(mask, channels=1) # Ou utiliser decode_jpeg si c'est un JPEG
    mask = tf.image.resize(mask, [256, 256])
    # mask = mask / 255.0 # Normaliser entre 0 et 1 si nécessaire
    return image, mask

def augment_image(image, mask):
    # Appliquer une rotation aléatoire, un décalage, etc.
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    if tf.random.uniform(()) > 0.5:
        dx, dy = 10, 10  # décalage souhaité
        image = tf.image.pad_to_bounding_box(image, dy, dx, 256 + dy, 256 + dx)
        mask = tf.image.pad_to_bounding_box(mask, dy, dx, 256 + dy, 256 + dx)
        image = tf.image.crop_to_bounding_box(image, 0, 0, 256, 256)
        mask = tf.image.crop_to_bounding_box(mask, 0, 0, 256, 256)

    if tf.random.uniform(()) > 0.5:
        zoom_ratio = 0.8  # supérieur à 1 pour zoom avant, inférieur à 1 pour zoom arrière
        new_width, new_height = int(256 * zoom_ratio), int(256 * zoom_ratio)
        image = tf.image.resize(image, [new_width, new_height])
        mask = tf.image.resize(mask, [new_width, new_height])
        image = tf.image.pad_to_bounding_box(image, (256 - new_width) // 2, (256 - new_height) // 2, 256, 256)
        mask = tf.image.pad_to_bounding_box(mask, (256 - new_width) // 2, (256 - new_height) // 2, 256, 256)

    return image, mask

def load_and_augment_segmentation(filepath, maskpath):
    image, mask = parse_function_segmentation(filepath, maskpath)
    image, mask = augment_image(image, mask)
    return image, mask

def create_dataset_segmentation(filepaths, maskpaths, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, maskpaths))
    if augment:
        dataset = dataset.map(load_and_augment_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(parse_function_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(buffer_size=1000).batch(32).repeat().prefetch(tf.data.AUTOTUNE)

train_dataset_segmentation = create_dataset_segmentation(train_filepaths, train_maskpaths, augment=True)
val_dataset_segmentation = create_dataset_segmentation(val_filepaths, val_maskpaths, augment=False)
test_dataset_segmentation = create_dataset_segmentation(test_filepaths, test_maskpaths, augment=False)

### Fonction pour visualiser l'image et le masque côte à côte

import matplotlib.pyplot as plt
import cv2

def display_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Supposons que le masque est en niveaux de gris

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")

    st.pyplot()

# Vérifier pour quelques paires aléatoires
import random

sample_indexes = random.sample(range(len(train_filepaths)), 5)  # choisir 5 paires aléatoirement

for index in sample_indexes:
    display_image_mask(train_filepaths[index], train_maskpaths[index])

### Chargement éventuel d'un modèle enregistré

custom_objects = {'DiceCoeff': DiceCoeff, 'LossDice': LossDice}
saved_segmenter = tf.keras.models.load_model(
    r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_unet_with_data_augmented\model_unet_with_data_augmented.h5', 
    custom_objects=custom_objects
)

#Prediction

n = len(test_filepaths)
batch_size = 32

steps_to_evaluate = n // batch_size

limited_dataset = test_dataset_segmentation.take(steps_to_evaluate)

### Evaluation de la précision du modèle sur le jeu de test
evaluation=saved_segmenter.evaluate(limited_dataset)
print('Précision du modèle sur le jeu de test :',np.round(evaluation[1],2),"\n",
      'Perte du modèle sur le jeu de test :',np.round(evaluation[0],2))

### Définition de fonctions de chargement et transformation
### des images ou masques à partir des chemins

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