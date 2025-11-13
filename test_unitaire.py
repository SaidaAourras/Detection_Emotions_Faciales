import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import pytest
# from detect_and_predict import detect_and_predict
from fastapi.testclient import TestClient
from backend.main import app
 
client = TestClient(app)

@pytest.fixture
def my_model():
    return tf.keras.models.load_model('CNN\CNN_model.keras')

def test_verify_model(my_model):
    assert os.path.exists('CNN\CNN_model.keras')
    model = my_model
    assert model 

def test_format_prediction():
    path_image = 'CNN/images/frt.jpg'
    with open(path_image , 'rb') as img:
        files = {"file": ("frt.jpg", img, "image/jpg")}
        response  = client.post('/predict_emotion' , files=files)
        
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data['emotion'], str) and isinstance(data['confidance'], float)


