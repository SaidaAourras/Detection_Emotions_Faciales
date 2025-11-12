import sys
import os
from fastapi import FastAPI , Depends , File , UploadFile
# from models import Create_Prediction
from sqlalchemy.orm import Session
# from PIL import Image
from database import Base , get_db , engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import CNN.detect_and_predict as cnn_predict
import numpy as np
import cv2
from models import Prediction
from sqlalchemy import func


app = FastAPI()

Base.metadata.create_all(engine)

@app.post('/predict_emotion' )
async def create_Prediction(file: UploadFile = File(...), db:Session = Depends(get_db)):
    
    # image = Image.open(file.file)
    # return {"hello" : image}
    file_bytes = await file.read()
        
    # octets to numpy
    numpy_img = np.frombuffer(file_bytes , np.uint8)
        
    # decoder avec openCv
    img = cv2.imdecode(numpy_img , cv2.IMREAD_COLOR)
        
    _ , emotions , scores = cnn_predict.detect_and_predict(img , 'CNN_model.keras')
    
    print(emotions)
    print(scores)
    
    for emotion , score in zip(emotions , scores):
        new_prediction = Prediction(
            emotion = emotion ,
            confidence = float(score) ,
            image_name = file.filename,
            created_at = func.current_date()
        )
        
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)
    
    # for score in scores :
    #     score = float(score)
    #     print(type(score))

@app.get('/history/{image_name}')
def get_history(image_name : str ,db:Session= Depends(get_db)):
    history_image = db.query(Prediction).where(Prediction.image_name == image_name).all()
    # for h in history_image:
    #     print(type(h))
    #     print(h)
    # print(type(history_image))
    # print(history_image)
    return {'history' : history_image}
    