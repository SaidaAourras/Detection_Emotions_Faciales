import os
from backend.main import app
import tensorflow as tf
import pytest
from backend.database import Base, engine
from fastapi.testclient import TestClient


client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def my_model():
    return tf.keras.models.load_model('CNN_model.keras')

def test_verify_model(my_model):
    assert os.path.exists('CNN_model.keras')
    model = my_model
    assert model 

def test_format_prediction():
    path_image = 'CNN/images/frt.jpg'
    with open(path_image , 'rb') as img:
        files = {"file": ("frt.jpg", img, "image/jpg")}
        response  = client.post('/predict_emotion' , files=files)
        
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data['emotion'], str) and isinstance(data['confidence'], float)
