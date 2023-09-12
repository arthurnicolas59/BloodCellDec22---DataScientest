import streamlit as st

st.set_page_config(
    page_title="Accueil ",
    page_icon="👋",
)


# Streamlit app title
st.title("Projet DataScientest : Classification de cellules sanguines - Dec 2022")
st.subheader('Ludovic Rochette - Aurélie Amira - Arthur Nicolas - Virginie Belleville')

#photo
col1, col2, col3 = st.columns([0.1,0.8,0.1])
with col1:
    st.write(' ')

with col2:
    st.image('streamlit_media/main.png')

with col3:
    st.write(' ')
