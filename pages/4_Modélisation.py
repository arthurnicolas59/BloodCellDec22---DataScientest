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
    page_title="Modèle Benchmark Images Brutes",
    #page_icon="👋",
)

#Titre
st.title("Modèle Benchmark Images Brutes")


st.set_option('deprecation.showPyplotGlobalUse', False)
df=pd.read_csv('Dataframe\df_cleaned.csv')

tab1, tab2, tab3, tab4,tab5=st.tabs(["**Structure**","**Chargement**","**Callbacks**","**Matrice de confusion**","**Images mal prédites**"])

with tab1:
    st.markdown('''
            * Structure du modèle
            ''')
    #Chargement du modèle Benchmark images brutes
    # model_benchmark_images_brutes = tf.keras.models.load_model('Model\model_benchmark\model_benchmark_images_brutes.h5')
    model_benchmark_images_brutes = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_benchmark\model_benchmark_images_brutes.h5')

    ## Présentation du Modèle
    from io import StringIO
    import sys

    # Création d'un objet StringIO pour capturer la sortie de la fonction summary
    buffer = StringIO()
    sys.stdout = buffer
    model_benchmark_images_brutes.summary()
    sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

    summary_str = buffer.getvalue()

    # Afficher dans Streamlit
    st.text(summary_str)


with tab2:
    st.markdown('''
            * Séparation du jeu de données et chargement des données
            ''')
    # Diviser le dataframe en ensembles d'entraînement, de validation et de test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Génération des données d'entrainement et de test
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

    ### Affichage dans streamlit
    code = '''
    # Diviser le dataframe en ensembles d'entraînement, de validation et de test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Génération des données d'entrainement et de test
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
        shuffle = False
    )

    nb_img_train = train_set.samples
    nb_img_val = validation_set.samples
    nb_img_test = test_set.samples

    label_map = train_set.class_indices
    '''

    st.code(code, language='python')

    st.markdown('''
            * Vérification du chargement des images dans le dataset train
            ''')

    import matplotlib.pyplot as plt

    def visualize_sample(image):
        plt.figure(figsize=(5, 5))

        plt.imshow(image[0])
        plt.axis('off')
        plt.title("image")

        st.pyplot()

    # Prendre un échantillon du train_dataset
    for idx, image in enumerate(train_set):
        if idx >= 5:  # Si vous avez déjà visualisé 5 images, arrêtez la boucle.
            break
        visualize_sample(image[0])


    st.caption("le dataframe train est composé de {} images appartenant à {} classes".format(nb_img_train, len(train_set.class_indices)))
    st.caption("le dataframe validation est composé de {} images appartenant à {} classes".format(nb_img_val, len(validation_set.class_indices)))
    st.caption("le dataframe validation est composé de {} images appartenant à {} classes".format(nb_img_test, len(test_set.class_indices)))

with tab3:
    st.markdown('''
            * Séparation du jeu de données et chargement des données
                ''')
    ### Affichage dans streamlit
    code = '''
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', #métrique à controller
                                         min_delta = 0.01, # si au bout de 5 epochs la fonction de perte ne varie pas de 1%, que ce soit à la hausse ou à la baisse, on arrête au bout de 5 épochs
                                         patience=10,
                                         verbose = 0,
                                         mode='min',
                                         restore_best_weights=True)

    lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', #métrique à controller
                                         factor=0.1, #factor by which the learning rate will be reduced. new_lr = lr * factor.
                                         patience=5, #number of epochs with no improvement after which learning rate will be reduced.
                                         verbose=0,
                                         mode='min')
    '''

    st.code(code, language='python')



with tab4:
    st.markdown('''
                * Matrice de confusion
                ''')

    #Prediction Modele Benchmark images brutes
    pred_benchmark_images_brutes = model_benchmark_images_brutes.predict(test_set)
    y_pred_benchmark_images_brutes = tf.argmax(pred_benchmark_images_brutes, axis = 1)

    from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix, roc_auc_score, precision_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay, classification_report

    #Création d'une Matrice de confusion
    def plot_matrix(y_true, y_pred, label):
        cm = confusion_matrix(y_true.classes, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        print('Vous retrouverez la matrice de confusion du modèle ci-dessous:')
        plt.figure()
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot()

        st.write('\nVous retrouverez le rapport de classification du modèle ci-dessous:\n')
        st.write(classification_report(y_true.classes, y_pred, target_names=label))


    plot_matrix(y_true = test_set, y_pred = y_pred_benchmark_images_brutes,label =label_map)

with tab5:
    st.markdown('''
                * Erreur de prédiction du modèle
                ''')
    predicted_labels = y_pred_benchmark_images_brutes
    true_labels = np.argmax(test_set, axis=1)

    predicted_labels = np.round(y_pred_benchmark_images_brutes).astype(int).flatten()
    true_labels = test_set.astype(int)

    incorrect_indices = np.where(predicted_labels != true_labels)[0]

    import streamlit as st

    MAX_IMAGES = 10
    for index in incorrect_indices[:MAX_IMAGES]:
        st.image(test_set[index], caption=f"True label: {true_labels[index]}, Predicted: {predicted_labels[index]}", use_column_width=True)




