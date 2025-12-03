FROM python:3.10-slim

WORKDIR /app

# 1. Instalamos solo lo básico (curl es útil para debug, pero opcional)
# Quitamos toda la lista de 'libs' que te estaba dando error.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Actualizar pip
RUN pip install --upgrade pip

# 3. Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 4. Copiar el código
COPY . .

# 5. Exponer puerto
EXPOSE 8000

# 6. Ejecutar
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]