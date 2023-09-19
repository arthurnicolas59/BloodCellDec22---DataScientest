#Import librairies
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

st.set_page_config(
    page_title="Essai3",
    page_icon="📊",
)

#EDA
st.header("Essai3")

df=pd.read_csv('Dataframe\df_cleaned.csv')

# Mapping entre nameLabel et label
mapping = {target: idx for idx, target in enumerate(df.target.unique())}

# Ajouter la colonne 'label' en utilisant le mapping
df['target_numerique'] = df['target'].map(mapping)

df.sort_values('Path')

st.write(df)

# Récupérer la liste unique des catégories dans votre DataFrame
categories = df['target'].unique()

# Créer un selectbox pour choisir une catégorie
selected_category = st.selectbox('Choisissez une catégorie:', categories)

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

list_category = df.target.unique()

# Assurez-vous qu'il y a au maximum 8 catégories
if len(list_category) > 8:
    raise ValueError("Il y a plus de 8 catégories, ce qui dépasse le nombre d'emplacements d'images disponibles.")

# Création de la figure
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))

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
st.pyplot(fig)



