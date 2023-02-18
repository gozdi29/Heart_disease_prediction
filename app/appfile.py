import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import sklearn
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent
filename = 'knn_pipeline.pkl'
classifier = pickle.load(open(f'{BASE_DIR}/{filename}', 'rb'))

templates = Jinja2Templates(directory="app/templates")

response_dict = {0: 'does not have',
                 1: 'has'}


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})

@app.post('/predict', response_class=HTMLResponse)
async def predict_disease(request: Request,
                          Age: int = Form(),
                          Sex: str = Form(),
                          ChestPainType: str = Form(),
                          RestingBP: int = Form(),
                          Cholesterol: int = Form(),
                          FastingBS: int = Form(),
                          RestingECG: str = Form(),
                          MaxHR: int = Form(),
                          ExerciseAngina: str = Form(),
                          Oldpeak: float = Form(),
                          ST_Slope: str = Form(),
):
    data = {
        'Age':Age,
        'Sex':Sex,
        'ChestPainType':ChestPainType,
        'RestingBP':RestingBP,
        'Cholesterol':Cholesterol,
        'FastingBS':FastingBS,
        'RestingECG':RestingECG,
        'MaxHR':MaxHR,
        'ExerciseAngina':ExerciseAngina,
        'Oldpeak':Oldpeak,
        'ST_Slope':ST_Slope
    }
    
    print(data)
    data = pd.DataFrame(data, index=[0])
    prediction = classifier.predict(data)
    #return f'the patient {response_dict[prediction[0]]} heart disease'
    return templates.TemplateResponse('home.html', {'request': request,
                                                    'prediction_text': f'the patient {response_dict[prediction[0]]} heart disease'
    })

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    