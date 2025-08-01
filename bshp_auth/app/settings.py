from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DB_URL: str = 'mongodb://localhost:27017'
    TEST_MODE: bool = False

    AUTH_DATABASE_NAME: str = 'bshp_auth'
    TOKEN_EXPIRATION_DAYS: int = 30
    SECRET_KEY: str = '26224616TK9T5S9HYFSQU62RZ1708XX9XKEWQUYLR00TYZ1Y25V889RZ'
    ENCODING_ALHORYTHM: str = 'HS256'
    AUTH_ADMIN_PASSWORD: str = 'pwd'
    
    class Config:
        extra = 'allow'        
        env_file = "../../.env"

settings = Settings()

for key, value in settings.__dict__.items():
    if not key.startswith('_'):
        globals()[key] = value  
