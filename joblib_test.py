import pandas as pd
import pickle
from pathlib import Path
import sklearn 

BASE_DIR = Path(__file__).resolve(strict=True).parent
filename = 'knn_pipeline.pkl'
classifier = pickle.load(open(f"{BASE_DIR}/app/{filename}", 'rb'))

dictionary = {
  "Age": 40,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 140,
  "Cholesterol": 289,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 172,
  "ExerciseAngina": "N",
  "Oldpeak": 0.0,
  "ST_Slope": "Up"
}

data = pd.DataFrame(dictionary, index=[0])
print(data)

pred = classifier.predict(data)
print(pred)

