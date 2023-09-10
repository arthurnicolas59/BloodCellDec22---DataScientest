import streamlit as st

# Streamlit app title
st.title("Projet DataScientest : Classification de cellules sanguines - Dec 2022")
st.subheader('Ludovic Rochette - Aurélie Amira - Arthur Nicolas - Virginie Belleville')



# Add a text element
st.write(
    """
    L’objectif du projet de modélisation est de réussir à identifier, à partir des images de cellules, leur type et 
    catégorie d’appartenance. Il s’agit donc d’un projet de classification multiclasses. \n
    Nous sommes partis des données récoltées par la clinique de Barcelone pour bâtir le modèle de reconnaissance des cellules.
    17 092 images de cellules normales ont été recensées grâce au CellaVision - DM96 du laboratoire de la clinique de Barcelone, 
    entre les années 2016 et 2019. Les patients étaient absolument sains, sans infection ou maladie, ni traitement 
    pharmaceutique au moment des prises de sang. \n
    Les données sont organisées en 8 groupes, en fonction du type de cellule observée.
    """
)

st.subheader('Les Globules Blancs')
globules_blanc_path =[
    'streamlit_media/neutrophil.png',
    'streamlit_media/eosinophil.png',
    'streamlit_media/basophil.png',
    'streamlit_media/lymphocyte.png',
    'streamlit_media/monocyte.png',
    'streamlit_media/granu.png',
]

globules_blanc_text = [
    'Neutrophiles : les plus abondants dans le sang, luttent contre les infections bactériennes',
    'Éosinophiles : impliqués dans la réponse immunitaire des parasites et réactions allergiques.',
    'Basophiles :  impliqués également en cas de réactions allergiques et libèrent des substances telles que l’histamine.',
    'Lymphocytes :  jouent un rôle central dans l’immunité adaptative, ils comprennent les Lymphocytes B (production d’anti corps) et les lymphocytes T (reconnaissance et destruction des cellules infectées)',
    'Monocytes : en charge de la phagocytose des débris cellulaires et des agents pathogènes',
    'Granulocytes Immatures (promyélocytes, myélocytes, métamyélocytes) : globules blancs immatures qui sont produits en grandes quantités en cas d’infections graves, type leucémie.'
]

for text, path in zip(globules_blanc_text, globules_blanc_path):
    st.write(text)
    st.image(path)

#Globules rouges
st.subheader('Les Globules Rouges')
st.write('Erythroblastes : globule rouge immatures')
st.image('streamlit_media/erythroblast.png')

#Platelet
st.subheader('Les Plaquettes')
st.write('Plaquettes (thrombocytes) : petites cellules fragmentées impliquées dans la coagulation du sang et la formation de caillots.')
st.image('streamlit_media/platelet.png')


#EDA
st.subheader("EDA")
st.write('Répartition des images par classe')
st.image('streamlit_media/edaclass.png')

st.write('Affichage de l’image moyenne pour un échantillon de 1000 images par catégorie')
st.image('streamlit_media/edameanimage.png')