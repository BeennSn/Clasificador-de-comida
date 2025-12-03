import io
import json
import uvicorn
from contextlib import asynccontextmanager
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, HTTPException

from utils import load_model_and_classes, predict_from_bytes
from clients import OpenFoodFactsClient, NutritionProvider

# Load synonyms
try:
    with open('synonyms.json', 'r', encoding='utf-8') as f:
        _SYNONYMS = json.load(f)
except Exception:
    _SYNONYMS = None

MODEL_PATH = "./ckpt_finetuned.pt"
_client = OpenFoodFactsClient()
_provider = NutritionProvider(synonyms=_SYNONYMS)

model = None
classes = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classes, device
    try:
        print("Loading model...")
        model, classes, device = load_model_and_classes(MODEL_PATH)
        print(f"Model loaded successfully! Classes: {len(classes) if classes else 0}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        model, classes, device = None, [], None
    yield


app = FastAPI(title="Food Classifier + Nutrition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a cualquier página web conectarse
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Food Classifier + Nutrition API", "status": "running"}


@app.get("/health")
def health_check():
    try:
        return {
            "status": "ok", 
            "model_loaded": model is not None,
            "classes_count": len(classes) if classes else 0,
            "device": str(device) if device else None
        }
    except Exception as e:
        print(f"Health check error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, classes, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        preds = predict_from_bytes(content, model, classes, device, topk=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    nutrition = None
    if preds:
        labels = [p.get('label') for p in preds]
        nutrition = _provider.get_nutrition_for_labels(labels)

    return {"predictions": preds, "nutrition": nutrition}


@app.get("/nutrition")
def nutrition_lookup(q: str):
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    result = _client.get_best_nutriments(q)
    if not result:
        raise HTTPException(status_code=404, detail="No nutrition data found")
    return result


@app.get("/classes")
def list_classes():
    global classes
    if classes is None or len(classes) == 0:
        raise HTTPException(status_code=500, detail="Model/classes not loaded yet")
    return {"classes": classes}


@app.get("/off-search")
def off_search(q: str, page_size: int = 10):
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    data = _client.search(q, page_size=page_size)
    if not data:
        raise HTTPException(status_code=404, detail="No results from Open Food Facts")
    products = data.get('products', [])[:page_size]
    out = []
    for p in products:
        out.append({
            'product_name': p.get('product_name') or p.get('generic_name'),
            'brands': p.get('brands'),
            'code': p.get('code'),
            'nutriments': p.get('nutriments'),
            'serving_size': p.get('serving_size')
        })
    return {'query': q, 'count': len(out), 'products': out}


@app.get("/off-suggestions")
def off_suggestions(q: str, page_size: int = 20):
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    data = _client.search(q, page_size=page_size)
    if not data:
        raise HTTPException(status_code=404, detail="No results from Open Food Facts")
    products = data.get('products', [])
    seen = set()
    suggestions = []
    for p in products:
        name = p.get('product_name') or p.get('generic_name')
        brand = p.get('brands')
        key = (name or '').strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        suggestions.append({'name': name, 'brands': brand})
        if len(suggestions) >= page_size:
            break
    return {'query': q, 'suggestions': suggestions}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")