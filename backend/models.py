from backend.database import Base
from pydantic import BaseModel
from  sqlalchemy import Column , Integer , String , Float , Date 

# model
class Prediction(Base):
    __tablename__='predictions'
    
    id =  Column(Integer , primary_key=True)
    emotion = Column(String)
    confidence = Column(Float)
    image_name = Column(String)
    created_at = Column(Date)
    
# schema
class Create_Prediction(BaseModel):
    emotion : str
    confidence : float