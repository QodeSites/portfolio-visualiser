# app/services/database.py

import psycopg2
from app.config import Config
import logging

def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise e
