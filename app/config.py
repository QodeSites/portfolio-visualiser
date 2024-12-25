import os
from datetime import timedelta

class Config:
    # Determine environment
    ENV = os.environ.get('FLASK_ENV', 'development')
    IS_DEVELOPMENT = ENV == 'development'
    
    SECRET_KEY = os.environ.get('SECRET_KEY', 'm3d2sti4EFoYUI4VS8v20Lde3Oapd756')
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    SESSION_FILE_DIR = os.path.join(PROJECT_ROOT, 'flask_session')
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    SESSION_USE_SIGNER = True
    
    # Cookie settings - adjust based on environment
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'None'
    SESSION_COOKIE_NAME = 'session'
    SESSION_COOKIE_DOMAIN = None
    
    # Create session directory if it doesn't exist
    os.makedirs(SESSION_FILE_DIR, exist_ok=True)
    
    # Your existing database configuration
    POSTGRES_USER = os.environ.get('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'S@nket@123')
    POSTGRES_DB = os.environ.get('POSTGRES_DB', 'QodeInvestments')
    POSTGRES_HOST = os.environ.get('POSTGRES_HOST', '139.5.190.184')
    POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
    SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    CSV_DIR = os.path.join(PROJECT_ROOT, 'csv')