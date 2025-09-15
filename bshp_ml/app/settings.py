from pydantic_settings import BaseSettings
from pathlib import Path
import os

VERSION = '3.2.0.0'
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    DB_URL: str = 'mongodb://localhost:27017'
    SOURCE_FOLDER: Path = BASE_DIR / 'data/source'
    TEMP_FOLDER: Path = BASE_DIR / 'data/temp'
    MODEL_FOLDER: Path = BASE_DIR / 'models'
    TEST_MODE: bool = False
    AUTH_SERVICE_URL: str = ''
    THREAD_COUNT: int = 4
    USE_DETAILED_LOG: bool = True
    class Config:
        env_file = "../../.env"
        extra = 'allow'
        
settings = Settings()

for key, value in settings.__dict__.items():
    if not key.startswith('_'):
        globals()[key] = value  



