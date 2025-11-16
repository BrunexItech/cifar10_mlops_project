from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_saver import ModelSaver

app = FastAPI(title="CIFAR-10 Classification API", version="1.0.0")

model = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']

def load_production_model():
    global model
    try:
        saver = ModelSaver()
        model = saver.load_model("cifar10_cnn")
        print("Production model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_production_model()

@app.get("/")
async def root():
    return {"message": "CIFAR-10 Classification API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((32, 32))
        
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            "predicted_class": class_names[predicted_class_idx],
            "confidence": confidence,
            "class_index": int(predicted_class_idx)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
