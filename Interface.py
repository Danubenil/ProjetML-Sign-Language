import streamlit as st
import cv2
import os
import numpy as np 
import pickle
import models
import transforms
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torch.nn
import torchvision.models as models
from PIL import Image
PATH_TO_MODELS = "./models"

class Interface : 

    def __init__(self, cheminStockage, dimensions): 
        self.cheminStockage = cheminStockage
        self.dimensions = dimensions
        self.model = None

    def prepare_images(X, cv2 = True):
        """
        X : dataset à traiter
        img : représente l'image qu'on souhaite pré-traiter, type : ndarray
        scalar : Objet de transformation d'image pour redimensionner l'image (par défaut, redimensionne à 224x224)
        normalizer : Objet de transformation d'image pour normaliser les valeurs des pixels (par défaut, normalise selon les valeurs de moyenne et d'écart type spécifiées pour ImageNet)
        to_tensor : Objet de transformation d'image pour convertir l'image en un tenseur (par défaut, convertit l'image en un tenseur)
        """
        RESNET_MEAN = [0.485, 0.456, 0.406]
        RESNET_STD = [0.229, 0.224, 0.225]
        RESNET_SHAPE = (224, 224)   
    # création du réseau de neurones pré-entraîné
        model = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
    # Récupération de la couche d'embedding (features) du réseau
        layer = model._modules.get("avgpool")
        model.eval()
        scaler = transforms.Resize(RESNET_SHAPE)
        normalizer = transforms.Normalize(mean = RESNET_MEAN, std = RESNET_STD)
        to_tensor = transforms.ToTensor()
        embeddings = []
        for img in tqdm(X):
            if cv2:
                img = transforms.ToPILImage()(img)
            img = Variable(normalizer(to_tensor(scaler(img))).unsqueeze(0))
            embedding = torch.zeros(1, 512, 1, 1)
            def copy_data(m, i, o):
                embedding.copy_(o.data)
            h = layer.register_forward_hook(copy_data)
            model(img)
            h.remove()
            embeddings.append(embedding.flatten())
        return np.array([embedding.numpy() for embedding in embeddings])

    def interface(self):
        st.title("SIGNAVERSE")  # Titre de l'application

        st.sidebar.header("         Paramètres         ")  # Titre de la barre latérale pour les paramètres

        # Bouton pour insérer une image

        
        model = st.sidebar.selectbox("model", ["Knn", "Regression logistique", "SVM", "Random forest", "Multilayer perceptron"])  # Sélection du format vidéo
        if model == "Knn":
            with open(os.path.join(PATH_TO_MODELS, "knn"), "rb") as f:
                self.model = pickle.load(f)
        elif model == "Regression logistique":
            with open(os.path.join(PATH_TO_MODELS, "reg_log"), "rb") as f:
                self.model = pickle.load(f)
        elif model == "SVM":
            with open(os.path.join(PATH_TO_MODELS, "SVM"), "rb") as f:
                self.model = pickle.load(f)
        elif model == "Random forest":
            with open(os.path.join(PATH_TO_MODELS, "RandomForest"), "rb") as f:
                self.model = pickle.load(f)
        elif model == "Multilayer perceptron":
            with open(os.path.join(PATH_TO_MODELS, "MLP"), "rb") as f:
                self.model = pickle.load(f)
        image_uploaded = st.sidebar.file_uploader("Insérer une image", type=["jpg", "jpeg", "png"])

        if image_uploaded:
            # Afficher l'image
            st.image(image_uploaded, caption='Image insérée', use_column_width=True)

            # Conversion de l'image en tableau numpy pour le traitement
            img_array = np.array(Image.open(image_uploaded))

            print("Array : " , img_array)
            # Traitement de l'image avec le modèle
            img_array = Interface.prepare_images([img_array])
            prediction = self.model.predict(img_array)[0]
            #prediction = 1
            # Affichage de la prédiction
            st.write("Résultat de la prédiction :", prediction)


        # Bouton pour capturer une photo à partir de la webcam
        if st.sidebar.button("Capturer une photo"):
            # Capture d'une image à partir de la webcam
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                # Affichage de l'image capturée
                st.image(frame, caption='Photo prise', channels='BGR')

                # Enregistrement de l'image capturée
                nom_fichier = "photo_capturee.png"
                chemin_fichier = os.path.join(self.cheminStockage, nom_fichier)
                cv2.imwrite(chemin_fichier, frame)

                # Traitement de l'image avec le modèle
                prediction = self.model.predict(frame)

                # Affichage de la prédiction
                st.write("Résultat de la prédiction :", prediction)

            cap.release()  # Libération de la capture vidéo

        
        video_format = st.sidebar.selectbox("Format vidéo", ["avi", "mp4"])  # Sélection du format vidéo
        max_duration = st.sidebar.slider("Durée maximale (secondes)", 60, 240, 10)  # Sélection de la durée maximale

        st.sidebar.markdown("---")  # Ligne de séparation dans la barre latérale

        recording_state = st.empty()  # Élément d'interface utilisateur pour afficher l'état de l'enregistrement
        record_button = st.sidebar.button("Enregistrer une vidéo")  # Bouton pour démarrer l'enregistrement

        if record_button:  # Si le bouton d'enregistrement est cliqué
            st.session_state["recording"] = True  # Définition de l'état d'enregistrement dans la session

        if "recording" not in st.session_state:  # Si l'état d'enregistrement n'existe pas dans la session
            st.session_state["recording"] = False  # Initialisation de l'état d'enregistrement à False

        if st.session_state.recording:  # Si l'enregistrement est en cours
            recording_state.info("En train d'enregistrer...")  # Affichage de l'état d'enregistrement

            video_frames = []  # Liste pour stocker les frames vidéo
            duration = 0  # Initialisation de la durée d'enregistrement
            cap = cv2.VideoCapture(0)  # Capture vidéo à partir de la webcam

            while duration < max_duration:  # Tant que la durée d'enregistrement est inférieure à la durée maximale
                ret, frame = cap.read()  # Lecture d'une frame vidéo
                if not ret:  # Si la lecture échoue
                    break
                video_frames.append(frame)  # Ajout de la frame à la liste
                duration += 1  # Incrémentation de la durée

            cap.release()  # Libération de la capture vidéo

            recording_state.empty()  # Suppression de l'état d'enregistrement

            st.session_state.pop("recording")  # Suppression de l'état d'enregistrement de la session

            self.cheminVideo = "recorded_video." + video_format  # Nom du fichier vidéo enregistré
            if video_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec pour le format AVI
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec pour le format MP4
            out = cv2.VideoWriter(self.cheminVideo, fourcc, 20.0, (frame.shape[1], frame.shape[0]))  # Création du writer vidéo

            for frame in video_frames:  # Pour chaque frame dans la liste
                out.write(frame)  # Écriture de la frame dans la vidéo

            out.release()  # Fermeture du writer vidéo

            st.success(f"La vidéo a été enregistrée sous {self.cheminVideo}.")  # Affichage d'un message de succès

            st.markdown("---")  # Ligne de séparation dans l'interface principale
            st.header("Traduction")  # Titre pour la section de récupération de la vidéo
            st.write("Analyse et traduction en cours...")
            predictions = self.extraction_et_traduction()
            if predictions:
                st.write("Résultats de la traduction :")
                for prediction in predictions:
                    st.write(prediction)
            else:
                st.write("Aucune prédiction disponible.")

    def extraction_et_traduction(self):
        # Lecture de la vidéo
        video_capture = cv2.VideoCapture(self.cheminVideo)

        # Liste pour stocker les prédictions
        predictions = []
        derniere_prediction = None

        # Initialiser le compteur de frames
        numero_capture = 0

        # Parcours de toutes les images dans la vidéo
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Enregistrer la frame dans un répertoire
            nom_fichier = "image{}.png".format(numero_capture)
            chemin_fichier = os.path.join(self.cheminStockage, nom_fichier)
            
            # Redimensionner
            hauteur, largeur = self.dimensions
            image_redimensionnee = cv2.resize(frame, (largeur, hauteur))

            # Stockage
            cv2.imwrite(chemin_fichier, image_redimensionnee)

            # Passer l'image au modèle pour prédire la classe
            prediction = self.modele.predict(" ")

            # Supprimer l'image après l'avoir traduite

            # Vérifier si la prédiction est différente de la précédente
            if prediction != derniere_prediction:
                predictions.append(prediction)
                derniere_prediction = prediction

            # Incrémenter le compteur de frames
            numero_capture += 1

        return predictions

    
def main(): 
    i = Interface("./Images",(240,240))
    i.interface()

main()