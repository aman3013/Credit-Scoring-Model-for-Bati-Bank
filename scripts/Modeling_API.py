from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained model
with open('Credit_Scoring_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the input data model based on your features
class CreditScoringInput(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float
    MeanAmount: float
    StdAmount: float
    AvgTransactionHour: float
    AvgTransactionDay: float
    AvgTransactionMonth: float
    AvgTransactionYear: float
    TransactionVolatility_binned_WoE: float
    MonetaryAmount_binned_WoE: float
    NetCashFlow_binned_WoE: float
    DebitCreditRatio_binned_WoE: float
    logreg_risk_probability: float
    rf_risk_probability: float

@app.post("/predict")
async def predict(input_data: CreditScoringInput):
    try:
        # Convert input data to a numpy array in the correct order
        features = np.array([
            input_data.Recency,
            input_data.Frequency,
            input_data.Monetary,
            input_data.MeanAmount,
            input_data.StdAmount,
            input_data.AvgTransactionHour,
            input_data.AvgTransactionDay,
            input_data.AvgTransactionMonth,
            input_data.AvgTransactionYear,
            input_data.TransactionVolatility_binned_WoE,
            input_data.MonetaryAmount_binned_WoE,
            input_data.NetCashFlow_binned_WoE,
            input_data.DebitCreditRatio_binned_WoE,
            input_data.logreg_risk_probability,
            input_data.rf_risk_probability
        ]).reshape(1, -1)
        
        # Make predictions
        prediction = model.predict(features)
        
        # Get probabilities if available
        probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        # Map prediction to credit score and rating
        credit_score = 300 if prediction[0] == 1 else 740
        rating = "Poor" if prediction[0] == 1 else "Very Good"
        
        # Return the prediction, credit score, and rating
        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "credit_score": credit_score,
            "rating": rating
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)