import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle

# Modeli yükle
with open('new_model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Scaler'ı yükle (model eğitimi sırasında kullanılan scaler)
scaler = pickle.load(open('scaler.pkl', 'rb'))

def covid_tahmini_yap(model, image, scaler):
    # Resmi gri tonlamalıya çevir ve yeniden boyutlandır
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image).flatten().reshape(1, -1)
    
    # Veriyi ölçeklendir
    image = scaler.transform(image)
    
    # Tahmin yap
    tahmin = model.predict(image)
    
    return tahmin[0]

st.title("COVID-19 Resim Sınıflandırma")
st.write("Yüklediğiniz resmin COVID-19 olup olmadığını tahmin eden bir model.")

uploaded_file = st.file_uploader("Bir göğüs röntgeni yükleyin...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Resim.', use_column_width=True)
    st.write("")
    st.write("Tahmin ediliyor...")

    label = covid_tahmini_yap(model, image, scaler)

    if label == 0:
        st.write("Sonuç: Resimde COVID-19 var.")
    else:
        st.write("Sonuç: Resimde COVID-19 yok.")
