from sqlalchemy.orm import declarative_base , sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

# variables d'environement
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DATABASE = os.getenv('DATABASE')


DATABASE_URL = f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
print(DATABASE_URL)

engine = create_engine(DATABASE_URL)

Base = declarative_base()

session = sessionmaker(autocommit=False ,autoflush=False, bind=engine)


def get_db():
    
    db = session()
    try:
        yield db
    finally:
        db.close()
        
    





