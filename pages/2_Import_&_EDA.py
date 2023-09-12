import streamlit as st

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
)

#EDA
st.subheader("EDA")
st.write('Répartition des images par classe')
st.image('streamlit_media/edaclass.png')

st.write('Affichage de l’image moyenne pour un échantillon de 1000 images par catégorie')
st.image('streamlit_media/edameanimage.png')