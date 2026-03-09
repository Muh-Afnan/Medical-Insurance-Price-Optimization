from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.predict import InsurancePredictor

app = FastAPI(title="Medical Insurance Price Predictor API")

# Initialize predictor
try:
    predictor = InsurancePredictor(model_path='models/insurance_model_pipeline.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Insurance Price Predictor API", "status": "Ready"}

@app.post("/predict")
def predict_single(data: InsuranceInput):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_dict = data.dict()
        prediction = predictor.predict(input_dict)
        return {
            "prediction": float(prediction[0]),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Check for required columns
        required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        predictions = predictor.predict(df)
        df['predicted_charges'] = predictions
        
        # Return summary and predictions
        return {
            "total_rows": len(df),
            "predictions": df[['predicted_charges']].to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
