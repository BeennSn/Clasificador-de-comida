ejecución con python:

1.  **Clonar el repositorio:**
   
2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación:**
    ```bash
    uvicorn app:app --reload
    ```
ejecución con docker:

1. **Construir la imagen:**
    ```bash
    docker build -t nutri-food-app .
    ```
2. **Ejecutar el contenedor:**
    ```bash
    docker run -d -p 8000:8000 --name mi-nutri-ia nutri-food-app
    ```    
