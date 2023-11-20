# endpoint.py
from fastapi import FastAPI
from pydantic import BaseModel
from prediction import predictions
import uvicorn 
app = FastAPI()

class AccidentAnalysisRequest(BaseModel):
    year: int
    month: int
    category: str = None  
    accident_type: str = None

@app.post("/predictions")
async def get_prediction(data: AccidentAnalysisRequest):
    # Prepare the data to pass to the predictions function
    prediction_result = predictions(data.category, data.accident_type, data.year, data.month)
    return {"prediction": prediction_result}

if __name__ == "__main__":         
    uvicorn.run(app, host="127.0.0.1", port=8000)
