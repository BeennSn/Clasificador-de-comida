FROM python:3.10-slim

WORKDIR /app

# Instala dependencias del sistema necesarias para PyTorch y PIL
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primero para mejor cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto
COPY . .

# Crea directorio para cache si no existe
RUN mkdir -p /app/logs

# Expone el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]