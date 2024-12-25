
# app/routes/portfolio.py
import os
from flask import Blueprint, json, request, jsonify, session
import numpy as np
import pandas as pd
from app.services.analysis import calculate_portfolio_comparison, load_default_strategies
import logging

portfolio_bp = Blueprint('portfolio', __name__)

def numpy_json_encoder(obj):
    """
    Custom JSON encoder to handle numpy data types.
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                       np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

def safe_jsonify(data):
    """
    Safely convert data containing numpy types to JSON-serializable format.
    """
    return json.loads(json.dumps(data, default=numpy_json_encoder))

@portfolio_bp.route('/compare', methods=['POST'])
def compare_portfolios():
    try:
        data = request.json
        logging.info('Received comparison request data: %s', data)
        
        default_strategies_df = load_default_strategies()
        
        if default_strategies_df.empty:
            error_msg = "Default strategies data is unavailable."
            logging.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        # Get user strategies from session, with debug logging
        user_strategies = session.get('user_strategies', [])
        logging.info('Retrieved user strategies from session: %s records', len(user_strategies))
        
        if user_strategies:
            # Convert list of dictionaries to DataFrame
            df_user = pd.DataFrame(user_strategies)
            logging.info('User strategies columns: %s', df_user.columns.tolist())
            
            # Ensure date column is properly formatted in user strategies
            if 'date' in df_user.columns:
                df_user['date'] = pd.to_datetime(df_user['date'])
            
            combined_strategies_df = pd.concat([default_strategies_df, df_user], ignore_index=True)
            logging.info("Merged user-uploaded strategies with default strategies. Total rows: %d", len(combined_strategies_df))
        else:
            combined_strategies_df = default_strategies_df
            logging.info("No user-uploaded strategies found. Using default strategies only.")
        
        # Save combined strategies to CSV
        try:
            csv_filename = 'combined_strategies.csv'
            csv_path = os.path.join('data', csv_filename)
            os.makedirs('data', exist_ok=True)
            combined_strategies_df.to_csv(csv_path, index=False)
            logging.info(f"Successfully saved combined strategies to {csv_path}")
        except Exception as e:
            logging.warning(f"Failed to save combined strategies to CSV: {str(e)}")
        
        # Ensure date column is datetime
        combined_strategies_df['date'] = pd.to_datetime(combined_strategies_df['date'])
        
        logging.info('Combined strategies columns: %s', combined_strategies_df.columns.tolist())
        logging.info('Combined strategies shape: %s', combined_strategies_df.shape)
        
        results = calculate_portfolio_comparison(data, combined_strategies_df)
        return jsonify({'results': safe_jsonify(results)}), 200
        
    except Exception as e:
        logging.error(f"Error in compare_portfolios: {str(e)}")
        return jsonify({"error": str(e)}), 500