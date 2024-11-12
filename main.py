from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the saved model once when the app starts
model = joblib.load('xgb_regressor_model.joblib')
required_features = model.feature_names_in_

# Define the prediction endpoint with GET method
@app.get("/predict")
def predict_quantity(order_date: str, order_hour: int):
    try:
        # Process the input date to extract weekday and month
        date = pd.to_datetime(order_date, format='%Y-%m-%d')
        order_weekday = date.dayofweek  # Monday=0, Sunday=6
        order_month = date.month

        # Create a DataFrame with the features in the expected order
        input_data = pd.DataFrame([[order_weekday, order_month, order_hour]], columns=required_features)
        
        # Make the prediction
        predicted_quantity = model.predict(input_data)[0]
        
        # Convert to float for JSON serialization
        return {"predicted_quantity": float(predicted_quantity)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Test the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)