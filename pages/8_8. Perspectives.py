import streamlit as st

st.set_page_config(
    page_title="Perspectives",
    #page_icon="👋",
)

#Titre
st.title("Adaptation d'un code de détection de visage")

tab1, tab2,tab3=st.tabs(["**Objectifs**","**Implémentation**","**Résultats**"])
with tab1:
    col1,col2 =st.columns([0.55,0.45])
    with col1:
        st.write("")
    with col2:
        st.write("")
with tab2:
    st.write("")
with tab3:
    st.write("")