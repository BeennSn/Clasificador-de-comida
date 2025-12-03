import streamlit as st
import requests
from PIL import Image
import io

# Configuración de la página
st.set_page_config(page_title="Nutri-Food AI", page_icon="🥗", layout="centered")

# Título y Diseño
st.title("🥗 Nutri-Food AI")
st.write("Sube una foto de tu comida y la IA te dirá qué es y sus calorías.")

# Barra lateral para opciones
st.sidebar.header("Opciones")
confidence_threshold = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.5)

# Widget para subir imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Botón para analizar
    if st.button('🔍 Analizar Comida'):
        with st.spinner('Consultando a la IA y Base de Datos Nutricional...'):
            try:
                # 1. Preparar la imagen para enviarla a TU API (Docker)
                # Convertimos la imagen de vuelta a bytes para enviarla
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                
                # 2. Petición a tu API (Asegúrate de que Docker esté corriendo en el puerto 8000)
                response = requests.post("http://localhost:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- MOSTRAR RESULTADOS ---
                    
                    # A. Predicción Principal
                    main_pred = data['predictions'][0]
                    label = main_pred['label'].replace("_", " ").title()
                    conf = main_pred['confidence'] * 100
                    
                    st.success(f"¡Es **{label}**! (Certeza: {conf:.2f}%)")
                    
                    # B. Datos Nutricionales (Bonitos)
                    nutri = data.get('nutrition', {})
                    if nutri and 'result' in nutri:
                        info = nutri['result']['nutriments']
                        
                        st.subheader("📊 Información Nutricional (aprox. por 100g)")
                        
                        # Crear columnas para métricas tipo "Dashboard"
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Extraer datos con seguridad (si no existen pone 0)
                        kcal = info.get('energy-kcal_100g', 0)
                        carb = info.get('carbohydrates_100g', 0)
                        prot = info.get('proteins_100g', 0)
                        fat = info.get('fat_100g', 0)
                        
                        col1.metric("Calorías", f"{kcal} kcal")
                        col2.metric("Carbohidratos", f"{carb} g")
                        col3.metric("Proteínas", f"{prot} g")
                        col4.metric("Grasas", f"{fat} g")
                        
                        # Expandible con más detalles
                        with st.expander("Ver detalles completos"):
                            st.json(nutri)
                    else:
                        st.warning("No se encontró información nutricional exacta para este plato.")
                        
                else:
                    st.error(f"Error en la API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Ocurrió un error de conexión: {e}")
                st.info("Asegúrate de que tu contenedor Docker esté corriendo en el puerto 8000.")