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


# Trouver tous les chemins vers les fichiers qui finissent par .jpg
liste = glob.glob(r'C:\\Users\\lrochette\\BloodCellDec22---DataScientest\\Images\\PBC_dataset_normal_DIB\\*\\*.jpg')


# Création d'une liste comprenant les chemins d'accès aux images ainsi que le label de chaque image
liste = list(map(lambda x : [x, x.split('\\')[12].split('_')[0]], liste))
                                        
# Créer un DataFrame pandas
df = pd.DataFrame(liste, columns=['Path', 'target'])
df=df.replace(['SNE','BNE'],'NEUTROPHIL')
df=df.replace(['MY','MMY','PMY'],'IG')

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



