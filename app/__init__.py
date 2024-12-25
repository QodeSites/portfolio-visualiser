from flask import Flask
from flask_cors import CORS
from flask_session import Session
from .routes.portfolio import portfolio_bp
from .routes.upload import upload_bp
from .utils.logging_config import setup_logging
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    setup_logging()
    
    # Initialize Flask-Session
    Session(app)
    
    # Configure CORS with specific origins and proper credentials handling
    CORS(app, 
         resources={r"/api/*": {
             "origins": [
                 "https://qodeinvest.com",
                 "https://qodepreview.netlify.app",
                 "https://www.qodepreview.netlify.app",
                 "http://localhost:5173",
                 "http://localhost:3000",
                 "http://192.168.0.106:3000",
                 "http://192.168.0.106:5080"  # Add your API server origin
             ],
             "allow_credentials": True,  # Enable credentials
             "expose_headers": ["Set-Cookie"],  # Expose Set-Cookie header
             "supports_credentials": True,  # Legacy support
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
         }},
         supports_credentials=True  # Global credential support
    )
    
    # Register blueprints
    app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    
    return app