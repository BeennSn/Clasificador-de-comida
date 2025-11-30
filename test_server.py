from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "test from docker"}

@app.get("/health")
def health():
    return {"status": "ok", "source": "docker"}

# No necesitamos el if __name__ porque usaremos uvicorn directamente