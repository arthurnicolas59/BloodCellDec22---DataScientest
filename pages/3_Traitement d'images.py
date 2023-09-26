import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import PIL.Image
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import glob
import os
import streamlit as st
import cv2 as cv

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Traitement d'images",
    page_icon="📊",
)

#EDA
st.header("Traitement d'images")

# Chargement d'une image test

df=pd.read_csv('Dataframe/df_cleaned.csv')

num = st.slider('**Sélectionnez une image à traiter**', 0, 17091, 8000)
st.write(f"vous avez sélectionné une image de la catégorie:", df['target'][num])

#affichage de l'image
img_test = df['Path'][num]
img_test = cv.imread(img_test)
img_RGB = cv.cvtColor(img_test, cv.COLOR_BGR2RGB)
img_grayscale = cv.cvtColor(img_test, cv.COLOR_RGB2GRAY)
img_YUV = cv.cvtColor(img_test,cv.COLOR_BGR2YUV)

col1,col2,col3=st.columns([0.3,0.4,0.3])
with col1:
    st.write("")
with col2:
    fig, ax = plt.subplots(figsize=(1,1))
    ax.imshow(img_RGB)
    ax.axis('off')  

    st.pyplot(fig)

with col3:
    st.write("")

# Création d'un histogramme
def plot_histogram(init_img, convert_img):
    """Function allowing to display the initial
    and converted images according to a certain
    colorimetric format as well as the histogram
    of the latter.

    Parameters
    -------------------------------------------
    init_img : list
        init_img[0] = Title of the init image
        init_img[1] = Init openCV image
    convert_img : list
        convert_img[0] = Title of the converted
        convert_img[1] = converted openCV image
    -------------------------------------------
    """
    hist, bins = np.histogram(
                    convert_img[1].flatten(),
                    256, [0,256])
    # Distribution cumulative
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Histogramme
    fig = plt.figure(figsize=(6,12))

    plt.subplot(2, 1, 1)
    plt.imshow(convert_img[1])
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.plot(cdf_normalized,
             color='r', alpha=.7,
             linestyle='--')
    plt.hist(convert_img[1].flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title("Histogramme de l'image", color="#343434")
    
    st.pyplot()

tab1, tab2= st.tabs(["**Egalisation d'histogramme et filtrage**", "**Egalisation adaptative et filtrage**"])

with tab1:
    # st.subheader("Egalisation d'histogramme et filtrage")

    col1, col2,col3 = st.columns([0.33,0.33,0.34])
    with col1:
        st.write('Avant égalisation')
        plot_histogram(["RGB", img_RGB], ["YUV", img_YUV])

    with col2:
        st.write('Après égalisation')
        img_YUV[:,:,0] = cv.equalizeHist(img_YUV[:,:,0])
        img_equ = cv.cvtColor(img_YUV, cv.COLOR_YUV2RGB)
        plot_histogram(["RGB", img_RGB], ["Equalized", img_equ])

    with col3:
        st.write('Et filtrage non local des moyennes')
        # Application d'un filtrage non local des moyennes
        dst_img = cv.fastNlMeansDenoisingColored(
            src=img_equ,
            dst=None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21)

        # Show both img
        fig = plt.figure(figsize=(12,12))
        plt.subplot(1, 1, 1)
        plt.imshow(dst_img)
        plt.axis('off')
        st.pyplot()

    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:    
        st.image('streamlit_media/FlecheBas.png')
    with col3:
        st.write("")

    col1,col2=st.columns([0.8,0.2])
    with col1:
        st.write('**Projection 2D t-SNE**')
        st.image('streamlit_media/Projection2DimagesEgalisees.png')
    with col2:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.image('streamlit_media/NON.png')



with tab2:
        
        def plot_histogram(image, title, ax, color='black'):
            """Affiche l'histogramme d'une image."""
            hist = cv.calcHist([image], [0], None, [256], [0,256])
            ax.plot(hist, color=color)
            ax.set_xlim([0,256])
            ax.set_title(title)

        # Convertir l'image de RGB à LAB
        image_lab = cv.cvtColor(img_test, cv.COLOR_BGR2Lab)

        # Extraire le canal L
        l_channel, a_channel, b_channel = cv.split(image_lab)

        # Créer un objet CLAHE (les valeurs de clipLimit et tileGridSize peuvent être ajustées en fonction de vos besoins)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Appliquer l'égalisation adaptative au canal L
        cl_channel = clahe.apply(l_channel)

        # Fusionner les canaux pour obtenir l'image LAB avec égalisation adaptative
        merged_channels = cv.merge([cl_channel, a_channel, b_channel])

        # Convertir l'image de LAB à RGB
        image_equalized = cv.cvtColor(merged_channels, cv.COLOR_Lab2BGR)

        # Filtrage non local des moyennes
        image_filtered = cv.fastNlMeansDenoisingColored(
        src=image_equalized,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21)

        # Afficher les images et les histogrammes
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Afficher l'image originale
        axes[0, 0].imshow(cv.cvtColor(img_test, cv.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image Originale')
        axes[0, 0].axis('off')

        # Afficher l'image égalisée
        axes[0, 1].imshow(cv.cvtColor(image_equalized, cv.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image Egalisée')
        axes[0, 1].axis('off')

        # Afficher l'histogramme original
        plot_histogram(l_channel, 'Histogramme Image Orginale', axes[1, 0])

        # Afficher l'histogramme égalisé
        plot_histogram(cl_channel, 'Histogramme Image Egalisée', axes[1, 1])

        plt.tight_layout()
        st.pyplot()

        st.divider()
        col1,col2,col3=st.columns([0.4,0.2,0.4])
        with col1:
            st.write("")
        with col2:
            st.image('streamlit_media/FlecheBas.png')
        with col3:
            st.write("")

        col1,col2=st.columns([0.8,0.2])
        with col1:
            st.write('**Projection 2D t-SNE**')
            st.image('streamlit_media/Projection2DimagesEgalisationAdaptative.png')
        with col2:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.image('streamlit_media/NON.png')              

    