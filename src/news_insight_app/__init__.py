from flask import Flask
import os
from dotenv import load_dotenv

load_dotenv()  # no-op when env vars are already set (e.g. Heroku config vars)

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # Import and register blueprints
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app