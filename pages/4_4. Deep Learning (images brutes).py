import streamlit as st
import numpy as np
import os
import glob
import cv2
from keras import callbacks
import pandas as pd
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.applications import VGG16
from keras.layers import Multiply, Flatten, Dense, Input
from keras.models import Model
from keras.layers import BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import PIL.Image
from PIL import Image

st.set_page_config(
    page_title="Modèles entraînés sur images Brutes",
    #page_icon="👋",
)

#Titre
st.header("Modèles entraînés sur images Brutes")


st.set_option('deprecation.showPyplotGlobalUse', False)
df=pd.read_csv('Dataframe\df_cleaned.csv')

tab1, tab2, tab3, tab4,tab5=st.tabs(["**Structure**","**Callbacks**","**Matrice et rapport**","**Prédictions**","**Interprétabilité**"])

with tab1:
    col1,col2=st.columns([0.5,0.5])
    with col1 :
        st.markdown('''
            * CNN Benchmark
            ''')
        #Chargement du modèle Benchmark images brutes
        # model_benchmark_images_brutes = tf.keras.models.load_model('Model\model_benchmark\model_benchmark_images_brutes.h5')
        model_benchmark_images_brutes = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_benchmark\model_benchmark_images_brutes.h5')

        ## Présentation du Modèle
        # st.image('streamlit_media\plot_Model_Benchmark.png',width=400)
        from io import StringIO
        import sys

        # Création d'un objet StringIO pour capturer la sortie de la fonction summary
        buffer = StringIO()
        sys.stdout = buffer
        model_benchmark_images_brutes.summary()
        sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

        summary_str = buffer.getvalue()

        # Afficher dans Streamlit
        st.text(summary_str)
    with col2 :
        st.markdown('''
        * VGG16
        ''')
        #Chargement du modèle Benchmark images brutes
        # model_benchmark_images_brutes = tf.keras.models.load_model('Model\model_benchmark\model_benchmark_images_brutes.h5')
        model_VGG16_images_brutes = tf.keras.models.load_model(r'C:\Users\lrochette\Documents\Perso\DataScientist\BloodCellDec22---DataScientest\Model\model_vgg16_images_brutes\model_VGG16_images_brutes.h5')

        ## Présentation du Modèle
        # st.image('streamlit_media\plot_Model_Benchmark.png',width=400)
        from io import StringIO
        import sys

        # Création d'un objet StringIO pour capturer la sortie de la fonction summary
        buffer = StringIO()
        sys.stdout = buffer
        model_VGG16_images_brutes.summary()
        sys.stdout = sys.__stdout__  # Réinitialisez la sortie standard

        summary_str = buffer.getvalue()

        # Afficher dans Streamlit
        st.text(summary_str)



# with tab2:
#     st.markdown('''
#             * Séparation du jeu de données et chargement des données
#             ''')
    # Diviser le dataframe en ensembles d'entraînement, de validation et de test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Génération des données d'entrainement et de test
    datagen_train = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
    datagen_test = ImageDataGenerator(rescale = 1./255)

    batch_size = 32
    height  = 224 # taille adaptée à VGG16
    width = 224
    color = 3

    train_set = datagen_train.flow_from_dataframe(
        dataframe = train_df,
        directory=None,
        x_col='Path',
        y_col='target',
        target_size=(height, width),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset = "training"
    )

    validation_set = datagen_train.flow_from_dataframe(
        dataframe = train_df,
        directory=None,
        x_col='Path',
        y_col='target',
        target_size=(height, width),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        subset = "validation"
    )

    test_set = datagen_test.flow_from_dataframe(
        dataframe = test_df,
        directory=None,
        x_col='Path',
        y_col='target',
        target_size=(height, width),
        color_mode = 'rgb',
        classes = None,   # utilise y_col
        class_mode = 'categorical',
        batch_size = batch_size,
        shuffle = False)



    nb_img_train = train_set.samples
    nb_img_val = validation_set.samples
    nb_img_test = test_set.samples

    label_map = train_set.class_indices

    # ### Affichage dans streamlit
    # code = '''
    # # Diviser le dataframe en ensembles d'entraînement, de validation et de test
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # # Génération des données d'entrainement et de test
    # datagen_train = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
    # datagen_test = ImageDataGenerator(rescale = 1./255)

    # batch_size = 32
    # height  = 224 # taille adaptée à VGG16
    # width = 224
    # color = 3

    # train_set = datagen_train.flow_from_dataframe(
    #     dataframe = train_df,
    #     directory=None,
    #     x_col='Path',
    #     y_col='target',
    #     target_size=(height, width),
    #     color_mode='rgb',
    #     classes=None,
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     shuffle=True,
    #     subset = "training"
    # )

    # validation_set = datagen_train.flow_from_dataframe(
    #     dataframe = train_df,
    #     directory=None,
    #     x_col='Path',
    #     y_col='target',
    #     target_size=(height, width),
    #     color_mode='rgb',
    #     classes=None,
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     shuffle=False,
    #     subset = "validation"
    # )

    # test_set = datagen_test.flow_from_dataframe(
    #     dataframe = test_df,
    #     directory=None,
    #     x_col='Path',
    #     y_col='target',
    #     target_size=(height, width),
    #     color_mode = 'rgb',
    #     classes = None,   # utilise y_col
    #     class_mode = 'categorical',
    #     batch_size = batch_size,
    #     shuffle = False
    # )

    # nb_img_train = train_set.samples
    # nb_img_val = validation_set.samples
    # nb_img_test = test_set.samples

    # label_map = train_set.class_indices
    # '''

    # st.code(code, language='python')
    
    # st.markdown('''
    #          * Vérification du chargement des images dans le dataset train
    #         ''')
    
    # col1,col2,col3=st.columns([0.3,0.4,0.3])
    # with col1:
    #     st.write("")
    # with col2:
        
    #     import matplotlib.pyplot as plt

    #     def visualize_sample(image):
    #         plt.figure(figsize=(5, 5))

    #         plt.imshow(image[0])
    #         plt.axis('off')
            
    #         st.pyplot()

    #     # Prendre un échantillon du train_dataset
    #     for idx, image in enumerate(train_set):
    #         if idx >= 5:  # Si vous avez déjà visualisé 5 images, arrêtez la boucle.
    #             break
    #         visualize_sample(image[0])


    #     st.caption("le dataframe train est composé de {} images appartenant à {} classes".format(nb_img_train, len(train_set.class_indices)))
    #     st.caption("le dataframe validation est composé de {} images appartenant à {} classes".format(nb_img_val, len(validation_set.class_indices)))
    #     st.caption("le dataframe validation est composé de {} images appartenant à {} classes".format(nb_img_test, len(test_set.class_indices)))
    # with col3:
    #     st.write("")
with tab2:
    # st.markdown('''
    #         * Séparation du jeu de données et chargement des données
    #             ''')
    ### Affichage dans streamlit
    code = '''
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', #métrique à controller
                                         min_delta = 0.01, # si au bout de 5 epochs la fonction de perte ne varie pas de 1%, que ce soit à la hausse ou à la baisse, on arrête au bout de 5 épochs
                                         patience=5,
                                         verbose = 0,
                                         mode='min',
                                         restore_best_weights=True)

    lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', #métrique à controller
                                         factor=0.1, #factor by which the learning rate will be reduced. new_lr = lr * factor.
                                         patience=5, #number of epochs with no improvement after which learning rate will be reduced.
                                         verbose=0,
                                         mode='min')
    '''

    st.code(code, language='python')
with tab3:
    
    col1,col2,col3=st.columns([0.475,0.05,0.475])
    with col1:
        st.markdown('''
                * CNN Benchmark : Matrice de confusion
                ''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_brutes.png')
        st.markdown('''
                * CNN Benchmark : Rapport de classification
                    ''')
        # df=pd.read_csv('Dataframe\df_report_Benchmark_Images_brutes.csv')
        # # labels=['BA','EO','ERB','IG','LY','MO','NEU','PLA',]
        # st.dataframe(df)

        st.image('streamlit_media\Rapport_Classification_benchmark_images_brutes.png')
        # #Prediction Modele Benchmark images brutes
        # pred_benchmark_images_brutes = model_benchmark_images_brutes.predict(test_set)
        # y_pred_benchmark_images_brutes = tf.argmax(pred_benchmark_images_brutes, axis = 1)

        # from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix, roc_auc_score, precision_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay, classification_report

        # #Création d'une Matrice de confusion
        # def plot_matrix(y_true, y_pred, label):
        #     cm = confusion_matrix(y_true.classes, y_pred)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        #     print('Vous retrouverez la matrice de confusion du modèle ci-dessous:')
        #     plt.figure()
        #     disp.plot(cmap=plt.cm.Blues)
        #     st.pyplot()

        #     st.caption('\nVous retrouverez le rapport de classification du modèle ci-dessous:\n')
        #     st.caption(classification_report(y_true.classes, y_pred, target_names=label))


        # plot_matrix(y_true = test_set, y_pred = y_pred_benchmark_images_brutes,label =label_map)

    with col2:
        st.write("")
    with col3:
        st.markdown('''
                * VGG16 : Matrice de confusion
                \n''')
        st.image('streamlit_media\Matrice_confusion_VGG16_images_brutes.png')
        st.markdown('''
                * VGG16 : Rapport de classification
                \n''')
        
        # df=pd.read_csv('Dataframe\df_report_Benchmark_Images_brutes.csv')
        # # labels=['BA','EO','ERB','IG','LY','MO','NEU','PLA',]
        # st.dataframe(df)

        st.image('streamlit_media\Rapport_Classification_VGG16_images_brutes.png')
        # #Prediction Modele Benchmark images brutes
        # pred_benchmark_images_brutes = model_benchmark_images_brutes.predict(test_set)
        # y_pred_benchmark_images_brutes = tf.argmax(pred_benchmark_images_brutes, axis = 1)

        # from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix, roc_auc_score, precision_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay, classification_report

        # #Création d'une Matrice de confusion
        # def plot_matrix(y_true, y_pred, label):
        #     cm = confusion_matrix(y_true.classes, y_pred)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        #     print('Vous retrouverez la matrice de confusion du modèle ci-dessous:')
        #     plt.figure()
        #     disp.plot(cmap=plt.cm.Blues)
        #     st.pyplot()

        #     st.caption('\nVous retrouverez le rapport de classification du modèle ci-dessous:\n')
        #     st.caption(classification_report(y_true.classes, y_pred, target_names=label))


        # plot_matrix(y_true = test_set, y_pred = y_pred_benchmark_images_brutes,label =label_map)
    
    st.divider()
    col1,col2,col3=st.columns([0.475,0.05,0.475])

    with col1:
        st.subheader("Score de rappel plus faible pour la classe IG\n Attention aux faux négatifs\n pour l'incidence médicale que cela peut avoir")      
    with col2:
        st.write("")
    with col3:
        st.subheader("Excellent score, même pour la classe IG")
with tab4:
    st.markdown('''
                * Erreur de prédiction du modèle CNN Benchmark
                ''')
    
    df_interpretation=pd.read_csv('Dataframe\df_interpretation.csv')
    df_new = df_interpretation.loc[df_interpretation['target'] != df_interpretation['prediction']] #je crée un dataframe reprenant uniquement la catégorie i
    
    df_new=df_new.reset_index(drop=True)

    
    import tensorflow as tf
    def load_image(filepath,resize=(256,256)):
        # Charger l'information brute en mémoire
        im = tf.io.read_file(filepath)
        # Décoder l'information en un tensorflow RGB (3 channels).
        im = tf.io.decode_jpeg(im, channels=3)
        #Redimensionner l'image
        return tf.image.resize(im, size=resize)

    
    
    indexes = np.random.choice(len(df_new), size=8, replace=False)
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    
    for i, idx in enumerate(indexes):
        image = load_image(df_new['Path'].iloc[idx])
        image = tf.cast(image, dtype=tf.int32)
        
        # Conversion des index linéaires en index de grille 2D
        row = i // 4
        col = i % 4
        
        axs[row, col].imshow(image)
        axs[row, col].axis("off")
        axs[row, col].set_title(f"Target: {df_new['target'].iloc[idx]} : Prediction: {df_new['prediction'].iloc[idx]}")

    st.pyplot()

    st.divider()

    st.markdown('''
            * Bonne prédiction du modèle CNN Benchmark
            ''')
       
    df_true = df_interpretation.loc[df_interpretation['target'] == df_interpretation['prediction']] #je crée un dataframe reprenant uniquement la catégorie i
    
    df_true=df_true.reset_index(drop=True)

    unique_categories = df_true['target'].unique()

    # Vérification pour s'assurer qu'il y a 8 catégories uniques
    # if len(unique_categories) != 8:
    #     st.write("Attention : il n'y a pas exactement 8 catégories uniques dans le dataframe.")
    #     return

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))

    list_path=[]

    for i, category in enumerate(unique_categories):
        # Sélectionnez une image aléatoire de cette catégorie
        subset = df_true[df_true["target"] == category].sample(n=1)
        image_path = subset["Path"].values[0]
        list_path.append(image_path)
        
        image = load_image(image_path)
        image = tf.cast(image, dtype=tf.int32)
        
        # Conversion des index linéaires en index de grille 2D
        row = i // 4
        col = i % 4
        
        axs[row, col].imshow(image)
        axs[row, col].axis("off")
        axs[row, col].set_title(f"Category: {category}")

    st.pyplot()
with tab5:
    # from tensorflow.keras.preprocessing import image
    # import numpy as np
    # from lime.lime_image import LimeImageExplainer

    # # Charger l'image à partir du chemin d'accès et la prétraiter
    # def load_and_preprocess_image(image_path):
    #     img = image.load_img(image_path, target_size=(224, 224))
    #     img_array = image.img_to_array(img)
    #     img_array = np.expand_dims(img_array, axis=0)  # Convertir l'image en un batch de taille (1, height, width, channels)
    #     img_array /= 255.  # rescale
    #     return img_array[0]  # return the first image in the batch (our only image)

    # # Modifier la fonction pour obtenir une explication
    # def get_explanation(numpy_image, model):
    #     explainer = LimeImageExplainer(verbose=False)
    #     explanation = explainer.explain_instance(
    #         image=numpy_image,
    #         classifier_fn=model.predict,
    #         top_labels=1,
    #         num_samples=100
    #     )
    #     dict_explainer = {'Label': explanation}
    #     return dict_explainer
    
    # def plot_explainer(dict_explainer):
    #     from skimage.segmentation import mark_boundaries
    #     for cle, objet in dict_explainer.items():
    #         # st.caption("Analyse LIME pour une image de la catégorie {}".format(cle))
    #         temp_1, mask_1 = objet.get_image_and_mask(objet.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    #         temp_2, mask_2 = objet.get_image_and_mask(objet.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    #         plt.figure(figsize=(10,10))
    #         plt.subplot(121)
    #         plt.imshow(mark_boundaries(temp_1, mask_1))
    #         plt.title('SuperPixel pour cette catégorie')
    #         plt.axis('off')

    #         plt.subplot(122)
    #         plt.imshow(mark_boundaries(temp_2, mask_2))
    #         plt.title('Répartition des pixels positifs (vert) et négatifs (rouge)')
    #         plt.axis('off')
    #         st.pyplot()

    col1,col2,col3=st.columns([0.475,0.05,0.475])
    with col1:
        st.markdown('''
                * CNN Benchmark :
                ''')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_brutes).png')
        st.image('streamlit_media\Interpretabilite_Benchmark(Images_brutes)_2.png')


        # Chargement de l'image et obtention de l'explication
        # for path in list_path:
        
        #     image_path = path
        #     numpy_image = load_and_preprocess_image(image_path)
        #     dict_explainer = get_explanation(numpy_image, model_benchmark_images_brutes)

        #     # Affichage de l'explication
        #     plot_explainer(dict_explainer)
    with col2:
        st.write("")
    
    with col3:
        st.markdown('''
                * VGG16 :
                ''')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes).png')
        st.image('streamlit_media\Interpretabilite_VGG16(Images_brutes)_2.png')
        # # Chargement de l'image et obtention de l'explication
        # for path in list_path:
        
        #     image_path = path
        #     numpy_image = load_and_preprocess_image(image_path)
        #     dict_explainer = get_explanation(numpy_image, model_VGG16_images_brutes)

        #     # Affichage de l'explication
        #     plot_explainer(dict_explainer)


    st.divider()
    col1,col2,col3=st.columns([0.475,0.05,0.475])
    with col1:
        st.markdown('''
                    "CNN Benchmark" considère mal les zones de l'image correspondant aux cellules sanguines
                    ''')
    with col2:
        st.write("")
    with col3:
        st.markdown('''
            VGG16 considère mieux les zones de l'image correspondant aux cellules sanguines
            ''')
        


    st.divider()
    col1,col2,col3=st.columns([0.4,0.2,0.4])
    with col1:
        st.write("")
    with col2:
        st.image('streamlit_media/FlecheBas.png',width=100)
    with col3:
        st.write("")

    
    col1,col2,col3=st.columns([0.2,0.6,0.2])
    with col1:
        st.write("")
    with col2:
        st.subheader('''
            :red[Vers une segmentation d'image et création de masques ?]
            ''')
    with col3:
        st.write("")





