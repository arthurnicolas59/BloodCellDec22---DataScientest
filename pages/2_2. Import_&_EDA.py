import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(
    page_title="EDA",
    page_icon="📊",
)

#EDA
st.header("Analyse Exploratoire des Données")

tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs(["**Structure du dataset**", "**Dataframe**", "**Aperçu**","**Répartition par classe**","**Moyenne et écart type**","**Projection 2D**"])

with tab1:
   st.subheader("Structure du dataset")
   st.write('Les images téléchargées sont réparties en 8 sous dossiers')
   st.image('streamlit_media\Explorateur_fichiers.png', width=200)

with tab2:
   st.subheader("Dataframe")
   st.write('''Selon cette architecture, les images présentaient les chemins d'accès suivants, que nous avons enregistrés dans un dataframe''')
   df=pd.read_csv('Dataframe\df_cleaned.csv')
   df.sort_values('Path')
   st.dataframe(df)

with tab3:
    st.subheader("Aperçu")
    st.divider()
    st.markdown('''
                * :red[Echantillon alétoire : 1 image par catégorie]''')

    if st.button('Afficher un nouvel échantillon aléatoire'):
        list_category = df.target.unique()

        # Assurez-vous qu'il y a au maximum 8 catégories
        if len(list_category) > 8:
            raise ValueError("Il y a plus de 8 catégories, ce qui dépasse le nombre d'emplacements d'images disponibles.")

        # Création de la figure
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

        for i, category in enumerate(list_category):
            subset = df[df["target"] == category].sample(n=1)
            image_path = subset["Path"].values[0]

            # Charger et afficher l'image
            image = Image.open(image_path)
            ax = axs[i // 4, i % 4]  # Ajustement de l'indexation pour 2x4 grille
            ax.imshow(image)
            ax.set_title(category)
            ax.axis("off")

        plt.tight_layout()  # Pour un espacement approprié entre les sous-tracés
        st.pyplot()
    st.divider()

   # Récupérer la liste unique des catégories dans votre DataFrame
    st.markdown('''
                * :red[Echantillon de 9 images par catégorie]''')
    categories = df['target'].unique()

    # Créer un selectbox pour choisir une catégorie
    selected_category = st.selectbox('Choisissez une catégorie:', categories,key="1")

    subset = df[df["target"] == selected_category].sample(n=9)

    #création de la figure
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# boucle for
    for i, ax in enumerate(axs.flat):
        # chargement image
        img = Image.open(subset.iloc[i]["Path"])
        # Plot image
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"Image {i+1}")
        plt.suptitle("Catégorie visualisée : {}".format(selected_category), fontsize=20)

    # Visualisation
    st.pyplot(fig)

with tab4:
    st.subheader("Répartition par classe")

#################################
    import plotly.express as px

    # Triez les catégories target par décompte décroissant
    sorted_targets = df['target'].value_counts().index.tolist()

    # Utilisez plotly pour créer un histogramme avec les catégories triées
    fig = px.histogram(df, x='target', title='Histogramme répartition des images par classe', category_orders={"target": sorted_targets})

    # Affichez le graphique dans Streamlit
    st.plotly_chart(fig)
    ######################

with tab5:
    # Offrez à l'utilisateur une boîte de sélection pour choisir une catégorie
    selected_category = st.selectbox('Choisissez une catégorie', df.target.unique(),key="2")

    subset = df['Path'][df["target"] == selected_category].sample(n=1000)
    liste = []

    for path in subset:
        # Lecture du fichier
        im = tf.io.read_file(path)
        # On décode le fichier
        im = tf.image.decode_jpeg(im, channels=3)
        # On uniformise la taille des images
        im = tf.image.resize(im, size=(363, 360))
        # Ajout à la liste
        liste.append(im)

    liste = np.array(liste)

    col1, col2 = st.columns([0.5,0.5])
    with col1:
        st.subheader("Image moyenne")
        st.markdown('''
                * :red[Echantillon sélectif]''')

        image_moyenne = np.mean(liste, axis=0)

        # Création de la figure
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(image_moyenne/255)
        ax.set_title(selected_category)
        ax.axis('off')

        st.pyplot(fig)
        st.divider()                              

        st.markdown('''
            * :red[Affichage multiple]''')
        st.image('streamlit_media/edameanimage.png')

    with col2:

        st.subheader("Ecart type")
        st.markdown('''
                * :red[Echantillon sélectif]''')

        image_std = np.std(liste, axis=0)

        # Création de la figure
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(image_std/255)
        ax.set_title(selected_category)
        ax.axis('off')

        st.pyplot(fig)
        st.divider()

        st.markdown('''
            * :red[Affichage multiple]''')
        st.image('streamlit_media/edastdimage.png')

with tab6:
    col1, col2 = st.columns([0.5,0.5])

    with col1:
        st.markdown('''
            * :red[**Projection 2D des images brutes**]''')
        st.image('streamlit_media/Projection2DimagesBrutes.png')

    with col2:
        st.markdown('''
            * :red[**Superposition des vignettes sur la projection**]''')
        st.image('streamlit_media/Projection2DimagesBrutesVignettes.png')

    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:        
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    col1,col2,col3=st.columns([0.2,0.6,0.2])
    with col1:
        st.write("")
    with col2:
        st.subheader(''':red[Intérêt d'un traitement d'images ?]''')
    with col3:
        st.write("")