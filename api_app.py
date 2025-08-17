from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
from io import StringIO
import logging
import traceback
from pydantic import BaseModel
from typing import List
import os





# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour la réponse
class PredictionResult(BaseModel):
    id: int
    prediction: str
    probability: float
    features: dict
    image_url: str  # Ajout de l'URL de l'image

class StatsResult(BaseModel):
    total: int
    genuine: int
    fake: int
    genuine_percentage: float
    fake_percentage: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    stats: StatsResult

app = FastAPI(debug=True)

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Chargement du modèle et du scaler
try:
    model = joblib.load('random_forest_model.sav')
    scaler = joblib.load('scaler.sav')
    print(type(scaler))
    logger.info("Modèle et scaler chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle/scaler: {str(e)}")
    raise RuntimeError("Impossible de charger le modèle ou le scaler") from e

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        logger.info(f"Fichier reçu: {file.filename}, taille: {len(contents)} bytes")
        
        try:
            data = StringIO(contents.decode('utf-8'))
            df = pd.read_csv(data, sep=None, engine='python')
        except:
            data = StringIO(contents.decode('cp1252'))
            df = pd.read_csv(data, sep=None, engine='python')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Mapping des colonnes
        column_mapping = {
            'diagonal': ['diagonal', 'dingmail', 'diagonale'],
            'height_left': ['height_left', 'hauteur_gauche'],
            'height_right': ['height_right', 'hauteur_droite'],
            'margin_low': ['margin_low', 'margin_bow', 'marge_basse'],
            'margin_up': ['margin_up', 'marge_haute'],
            'length': ['length', 'longueur']
        }
        
        # Renommage des colonnes
        for standard_name, variants in column_mapping.items():
            for variant in variants:
                if variant in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                    break
        
        # Vérification des colonnes requises
        required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes requises manquantes: {missing_cols}"
            )
        
        # Conversion des données et remplissage des valeurs manquantes
        df = df[required_columns].apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.median())
        
        # Standardisation des données
        X_scaled = scaler.transform(df)
        
        # Prédictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Formatage des résultats
        results = []
        for i, (pred, prob, features) in enumerate(zip(predictions, probabilities, df.to_dict('records'))):
            results.append({
                "id": i + 1,
                "prediction": "Genuine" if pred else "Fake",
                "probability": float(prob[1] if pred else prob[0]),
                "features": features,
                "image_url": f"/images/{'vrai' if pred else 'faux'}.png"  # URL de l'image
            })
        
        # Statistiques
        genuine_count = int(sum(predictions))
        fake_count = int(len(predictions) - genuine_count)
        
        response = {
            "predictions": results,
            "stats": {
                "total": len(predictions),
                "genuine": genuine_count,
                "fake": fake_count,
                "genuine_percentage": round(genuine_count / len(predictions) * 100, 2),
                "fake_percentage": round(fake_count / len(predictions) * 100, 2)
            }
        }
        
        return convert_numpy_types(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du serveur"
        )


from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
async def root():
    return {"message": "API de détection de faux billets"}