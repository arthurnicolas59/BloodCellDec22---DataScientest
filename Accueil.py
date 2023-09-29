import streamlit as st

st.set_page_config(
    page_title="Accueil ",
    page_icon="👋",
)


# Streamlit app title
st.title("Projet DataScientest : Classification de cellules sanguines - Dec 2022")
col1,col2=st.columns([0.8,0.2])
with col1:
    st.subheader('Aurélie Amira - Virginie Belleville Bonnefoy Arthur Nicolas - Ludovic Rochette')
with col2 :
    st.write("")
#photo
col1, col2, col3 = st.columns([0.05,0.9,0.05])
with col1:
    st.write(' ')

with col2:
    st.image('streamlit_media/main.png')

with col3:
    st.write(' ')
