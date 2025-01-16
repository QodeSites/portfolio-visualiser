# app/routes/upload.py
from flask import Blueprint, request, current_app, jsonify, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
import traceback

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/check_session', methods=['GET'])
def check_session():
    logging.info("Current session contents: %s", dict(session))
    return jsonify({
        "session_contents": dict(session),
        "has_user_strategies": 'user_strategies' in session,
        "strategies_count": len(session.get('user_strategies', []))
    })

# app/routes/upload.py
@upload_bp.route('/upload_strategy', methods=['POST'])
def upload_strategy():
    try:
        logging.info("Handling file upload request")
        
        if 'file' not in request.files:
            logging.error("No file part in the request.")
            return jsonify({"error": "No file part in the request."}), 400
            
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file.")
            return jsonify({"error": "No selected file."}), 400
            
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.csv'):
            logging.error("Invalid file type. Only CSV files are allowed.")
            return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400
            
        # Create uploads directory if not exists
        upload_dir = os.path.join(current_app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file to the uploads directory
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        logging.info(f"File saved at {file_path}")
        
        # Attempt to read the CSV to extract column information
        try:
            df = pd.read_csv(file_path)
            columns = [col for col in df.columns.tolist() if col.lower() != 'date']
        except Exception as e:
            logging.error(f"Failed to read CSV file: {str(e)}")
            return jsonify({"error": "Failed to read CSV file."}), 400
            
        if not columns:
            logging.error("No strategy columns found in the uploaded CSV.")
            return jsonify({"error": "No strategy columns found in the uploaded CSV."}), 400
            
        return jsonify({
            "message": "File uploaded successfully.",
            "columns": columns,
            "rows_processed": len(df)
        }), 200
        
    except Exception as e:
        logging.error(f"Error in upload_strategy: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error."}), 500
