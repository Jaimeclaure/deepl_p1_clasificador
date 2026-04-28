import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

st.set_page_config(page_title="Clasificador de imagenes inteligente", page_icon="")

st.title("Proyecto de Deep Learning / Jaime Claure")
st.write("Predicción de Imagenes con redes convolucionales (CNN) y por el metodo de Aprendizaje por Transferencia (ResNet50). Sube una imagen de un lugar de interés y nuestro modelo de Deep Learning identificará de qué lugar se trata.")

# Seleccionar el modelo a utilizar
st.sidebar.title("Configuración")
model_choice = st.sidebar.selectbox(
    "Elige el modelo:",
    ("ResNet50", "CNN")
)

# Definir la ruta relativa del modelo según la elección
if model_choice == "ResNet50 (Transfer Learning)":
    model_path = "resnet_transfer_scripted.pt"
else:
    model_path = "cnn_scratch_scripted.pt"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    # Cargar el modelo en la CPU para inferencia local
    model = torch.jit.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model(model_path)

if model is None:
    st.error(f"No se encontró el modelo exportado en: {model_path}. Asegúrate de haber ejecutado todo el notebook y de que el archivo `.pt` exista en la misma carpeta que `app.py`.")
else:
    # Definir transformaciones estándar de validación
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Lista estática de las 51 clases para no depender del dataset localmente
    class_names = [
        "00.Haleakala_National_Park", "01.Mount_Rainier_National_Park", "02.Ljubljana_Castle", "03.Dead_Sea",
        "04.Wroclaws_Dwarves", "05.London_Olympic_Stadium", "06.Niagara_Falls", "07.Stonehenge",
        "08.Grand_Canyon", "09.Golden_Gate_Bridge", "10.Edinburgh_Castle", "11.Mount_Rushmore_National_Memorial",
        "12.Kantanagar_Temple", "13.Yellowstone_National_Park", "14.Terminal_Tower", "15.Central_Park",
        "16.Eiffel_Tower", "17.Changdeokgung", "18.Delicate_Arch", "19.Vienna_City_Hall", "20.Matterhorn",
        "21.Taj_Mahal", "22.Moscow_Raceway", "23.Externsteine", "24.Soreq_Cave", "25.Banff_National_Park",
        "26.Pont_du_Gard", "27.Seattle_Japanese_Garden", "28.Sydney_Harbour_Bridge", "29.Petronas_Towers",
        "30.Brooklyn_Bridge", "31.Washington_Monument", "32.Hanging_Temple", "33.Sydney_Opera_House",
        "34.Great_Barrier_Reef", "35.Monumento_a_la_Revolucion", "36.Badlands_National_Park", "37.Atomium",
        "38.Forth_Bridge", "39.Gateway_of_India", "40.Stockholm_City_Hall", "41.Machu_Picchu",
        "42.Death_Valley_National_Park", "43.Gullfoss_Falls", "44.Trevi_Fountain", "45.Temple_of_Heaven",
        "46.Great_Wall_of_China", "47.Prague_Astronomical_Clock", "48.Whitby_Abbey", "49.Temple_of_Olympian_Zeus",
        "50.Uriondo"
    ]

    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen a analizar', use_container_width=True)
        
        st.write("Analizando la imagen...")
        
        # Preprocesar
        img_tensor = val_transforms(image).unsqueeze(0)
        
        # Predecir
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            top_prob, top_catid = torch.topk(probabilities, 3)
            
        st.subheader("Resultados - Top 3 Predicciones:")
        for i in range(3):
            prob = top_prob[0][i].item() * 100
            class_idx = top_catid[0][i].item()
            class_name = class_names[class_idx]
            
            st.write(f"**{i+1}. {class_name}** ({prob:.2f}%)")
            st.progress(prob / 100.0)
