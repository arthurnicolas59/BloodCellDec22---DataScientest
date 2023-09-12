import streamlit as st

st.set_page_config(
    page_title="Le Projet",
    #page_icon="👋",
)

#Titre
st.title("Le Projet")


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
    col1, col2, col3 = st.columns([0.3,0.4,0.3])
    with col1:
        st.write(' ')

    with col2:
        st.image(path)

    with col3:
        st.write(' ')


#Globules rouges
st.subheader('Les Globules Rouges')
st.write('Erythroblastes : globule rouge immatures')
col1, col2, col3 = st.columns([0.3,0.4,0.3])
with col1:
        st.write(' ')
with col2:
    st.image('streamlit_media/erythroblast.png')
with col3:
    st.write(' ')


#Platelet
st.subheader('Les Plaquettes')
st.write('Plaquettes (thrombocytes) : petites cellules fragmentées impliquées dans la coagulation du sang et la formation de caillots.')
col1, col2, col3 = st.columns([0.3,0.4,0.3])
with col1:
        st.write(' ')
with col2:
    st.image('streamlit_media/platelet.png')
with col3:
    st.write(' ')

