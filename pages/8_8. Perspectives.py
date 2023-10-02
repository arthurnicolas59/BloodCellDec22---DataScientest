import streamlit as st

st.set_page_config(
    page_title="Perspectives",
    #page_icon="👋",
)

#Titre
st.title("Adaptation d'un code de détection de visage")

tab1, tab2,tab3,tab4=st.tabs(["**Objectifs**","**Implémentation**","**Résultats**","**Conclusion**"])
with tab1:
    st.markdown('''
                > Vers une finalité médicale : le frottis sanguin présente une foule de cellules
                ''')
    col1,col2 =st.columns([0.1,0.9])
    with col1:
        st.write("")
    with col2:
        st.markdown('''
                > 1° Exploiter un code de reconnaissance de visage dans une foule \n
                > 2° Le tester sur le dataset Barcelone \n
                > 3° Analyser les résultats
                        ''')
                        
with tab2:
    # st.image('streamlit_media\perspectives_implementation_1.png')
    st.image('streamlit_media\perspectives_implementation_2.png')
with tab3:
    col1, col2=st.columns([0.5,0.5])
    with col1 :
        st.image('streamlit_media\perspectives_resultats_2.png')
    with col2 :
        st.image('streamlit_media\perspectives_resultats_1.png')
    st.write("")

with tab4:
    st.markdown('''
                > Implémentation incomplète
                > Mais au vu des perspectives, ne faudrait-il pas envisager de travailler sur un dataset d'images représentant une foule de cellules ?
                ''')
    st.image('streamlit_media\perspectives_conclusion.png')