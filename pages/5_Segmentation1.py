import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2

st.set_page_config(
    page_title="Segmentation1",
    page_icon="📊",
)

#EDA
st.header("Segmentation1")

st.set_option('deprecation.showPyplotGlobalUse', False)
df=pd.read_csv('Dataframe\df_mask_SAM.csv')

st.write(df)
### Génération de masques à partir des images segmentées du jeu de données

### Premièrement, on récupère les chemins des dossiers contenant les images



path= df['maskpath']

plt.figure(figsize=(15,7))
plt.subplot(121)
example = cv2.imread(path[20])


# Affichage de l'image originale
plt.imshow(example, vmin=0, vmax=1)
plt.title("Image segmentée d'origine")
plt.axis('off')

st.write(path[20])
st.write(example.min(), example.max())

threshold=20
for fichier in path :
    image=cv2.imread(fichier)
    if len(image.shape)==3:

    ### La moyenne des pixels RGB est calculée et un seuil est défini pour séparer les pixels
    ### de la feuille et le reste.
    ### La valeur de 1 est attribué pour un pixel de la feuille, le pixel prend la valeur 0 sinon.

        img_mask=(image.mean(axis=2)>threshold).astype(int)
        cv2.imwrite(fichier,img_mask)


plt.subplot(122)
example2=cv2.imread(path[20],cv2.IMREAD_GRAYSCALE)
plt.imshow(example2*255,cmap='gray')
plt.title("Masque généré à partir de l'image segmentée")
plt.axis('off');

st.write(example2.min(), example2.max())
st.pyplot()

copied_path = 'Images/masques_SAM\lymphocyte/LY_521923_mask.png'
st.write(path[2] == copied_path)
