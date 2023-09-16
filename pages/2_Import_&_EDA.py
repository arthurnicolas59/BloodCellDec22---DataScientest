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

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
)

#EDA
st.subheader("EDA")
st.write('Répartition des images par classe')

path=r'C:\Users\lrochette\BloodCellDec22---DataScientest\Dataframe\df_cleaned.csv'

df = pd.read_csv(path)

st.write(df)



# Fonction pour afficher le countplot
st.set_option('deprecation.showPyplotGlobalUse', False)

order = df['target'].value_counts().index


fig, ax = plt.subplots()
sns.countplot(x='target', data=df, order=order, ax=ax)
st.pyplot(fig)

st.write('Affichage de l’image moyenne pour un échantillon de 1000 images par catégorie')
st.image('streamlit_media/edameanimage.png')


category = "IG"  # a modifier par la catégorie visée
subset = df[df["target"] == category].sample(n=9)

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
    plt.suptitle("Catégorie visualisée : {}".format(category), fontsize=20)


# Visualisation
st.pyplot(fig)
