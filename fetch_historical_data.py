from tqdm import tqdm
import requests
import pandas as pd
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_fetch.log')

# List of instruments with their tokens (unchanged)
INSTRUMENTS = [
    ('256265', 'NIFTY 50'),
    ('276745', 'BSE500'),
    ('290569', 'NIFTY MICROCAP250'),
    ('256777', 'NIFTY MIDCAP 100'),
    ('267273', 'NIFTY SMLCAP 250'),
    ('263433', 'NIFTY AUTO'),
    ('260105', 'NIFTY BANK'),
    ('257289', 'NIFTY COMMODITIES'),
    ('288777', 'NIFTY CONSR DURBL'),
    ('257545', 'NIFTY CONSUMPTION'),
    ('268297', 'NIFTY CPSE'),
    ('261641', 'NIFTY ENERGY'),
    ('261897', 'NIFTY FMCG'),
    ('288521', 'NIFTY HEALTHCARE'),
    ('261385', 'NIFTY INFRA'),
    ('259849', 'NIFTY IT'),
    ('263945', 'NIFTY MEDIA'),
    ('263689', 'NIFTY METAL'),
    ('262153', 'NIFTY MNC'),
    ('262409', 'NIFTY PHARMA'),
    ('262921', 'NIFTY PSU BANK'),
    ('271113', 'NIFTY PVT BANK'),
    ('261129', 'NIFTY REALTY')
]

KITE_API_URL = "https://api.kite.trade/instruments/historical"
HEADERS = {
    "X-Kite-Version": "3",
    "Authorization": "token cme7fe5bmnl0wfqa:jRTuMP6hXa6A2Z3wdhPLenvnJ74LjlTx"
}

# Use the URL-encoded password
DB_CONNECTION_STRING = "postgresql://postgres:S%40nket%40123@139.5.190.184:5432/QodeInvestments"

def fetch_historical_data(instrument_token, start_date, end_date):
    all_data = []
    current_start = start_date
    retry_count = 0
    max_retries = 3

    while current_start <= end_date:
        current_end = min(
            datetime.strptime(current_start, '%Y-%m-%d') + timedelta(days=365),
            datetime.strptime(end_date, '%Y-%m-%d')
        ).strftime('%Y-%m-%d')

        params = {
            "from": current_start,
            "to": current_end
        }

        try:
            logging.info(f"Fetching data for token {instrument_token} from {current_start} to {current_end}")
            response = requests.get(
                f"{KITE_API_URL}/{instrument_token}/day", 
                headers=HEADERS, 
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'candles' in data['data']:
                    all_data.extend(data['data']['candles'])
                    retry_count = 0  # Reset retry count on successful fetch
                else:
                    logging.warning(f"Unexpected response structure for {instrument_token}: {data}")
                    retry_count += 1
            else:
                logging.error(f"Failed to fetch data for {instrument_token}: {response.text}")
                retry_count += 1
            
            current_start = (
                datetime.strptime(current_end, '%Y-%m-%d') + timedelta(days=1)
            ).strftime('%Y-%m-%d')
            
            # Exponential backoff for retries
            time.sleep(0.5 * (2 ** retry_count))
        
        except Exception as e:
            logging.error(f"Error fetching data for {instrument_token}: {e}")
            retry_count += 1
        
        # Break if max retries exceeded
        if retry_count >= max_retries:
            logging.error(f"Max retries exceeded for {instrument_token}. Skipping.")
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=["date", "open", "high", "low", "close", "volume"])
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    
    logging.warning(f"No data collected for instrument token {instrument_token}")
    return pd.DataFrame()

def upsert_instrument_data(session, instrument_name, instrument_data):
    """
    Upsert data for a single instrument using SQLAlchemy ORM approach
    """
    from sqlalchemy import text
    
    try:
        # Prepare the upsert SQL
        upsert_sql = text("""
            INSERT INTO tblresearch (indices, nav, date, direction, net_change, updated_at)
            VALUES (:indices, :nav, :date, :direction, :net_change, :updated_at)
            ON CONFLICT (indices, date) 
            DO UPDATE SET 
                nav = EXCLUDED.nav,
                direction = EXCLUDED.direction,
                net_change = EXCLUDED.net_change,
                updated_at = EXCLUDED.updated_at
        """)
        
        # Execute the upsert for each record
        for _, row in instrument_data.iterrows():
            session.execute(upsert_sql, {
                'indices': instrument_name,
                'nav': row['close'],
                'date': row['date'],
                'direction': 'Neutral',
                'net_change': 0,
                'updated_at': datetime.now().date()
            })
        
        session.commit()
        logging.info(f"Successfully saved/updated data for {instrument_name}")
        return True
    
    except Exception as e:
        session.rollback()
        logging.error(f"Error upserting data for {instrument_name}: {e}")
        return False

def process_and_save_data(start_date='2010-01-01', end_date='2024-12-17'):
    # Create engine
    engine = sqlalchemy.create_engine(DB_CONNECTION_STRING)
    
    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)
    
    # Create a Session
    session = Session()
    
    failed_instruments = []
    
    try:
        for token, instrument_name in tqdm(INSTRUMENTS, desc="Fetching data"):
            logging.info(f"Processing {instrument_name}")
            
            # Fetch historical data
            historical_data = fetch_historical_data(token, start_date, end_date)
            
            if not historical_data.empty:
                # Rename columns to match the database schema
                db_data = historical_data.copy()
                
                # Attempt to upsert the data
                if not upsert_instrument_data(session, instrument_name, db_data):
                    failed_instruments.append(instrument_name)
            else:
                logging.warning(f"No data for {instrument_name}")
                failed_instruments.append(instrument_name)
            
            # Small delay between instruments
            time.sleep(1)
    
    except Exception as e:
        logging.error(f"Unexpected error in main process: {e}")
        session.rollback()
    
    finally:
        # Always close the session
        session.close()
    
    if failed_instruments:
        logging.error("Failed to fetch or save data for these instruments:")
        for instrument in failed_instruments:
            logging.error(instrument)

if __name__ == "__main__":
    process_and_save_data()
    print("Data fetching and saving completed. Check data_fetch.log for details.")