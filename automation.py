import schedule
import time
import os
import sys
import random
import dotenv
import pyotp
import urllib.parse
import logging
import requests
import gzip
import pickle

from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from kiteconnect import KiteConnect, KiteTicker
from urllib.parse import urlparse, parse_qs
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from tqdm import tqdm

# Load environment variables from .env (if present)
dotenv.load_dotenv()

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Configuration
DB_CONNECTION_STRING = "postgresql://postgres:S%40nket%40123@139.5.190.184:5432/QodeInvestments"
KITE_API_URL = "https://api.kite.trade/instruments/historical"

# Path where we will save instrument data
INST_PATH = "./instruments/"

def update_env_file(key, value, env_path="./.env"):
    """
    Update or add a key-value pair in the .env file.
    If the key exists, it will be replaced; otherwise, appended.
    """
    lines = []
    key_found = False
    
    # Read existing lines, if the .env file exists
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
    
    # Process lines
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
    
    # If key not found, append
    if not key_found:
        lines.append(f"{key}={value}\n")
    
    # Write lines back
    with open(env_path, 'w') as f:
        f.writelines(lines)

def get_auth_token():
    """Get authentication token using Selenium"""
    AC = {
        'api_key': 'cme7fe5bmnl0wfqa',
        'api_secret': 'no15x3wpugb63yt6ju1jcuzmjo0zp5vc',
        'USER_ID': 'UIK957',
        'PASS': 'Qode@pms123',
        'pin': 'UWXLCMWDU46UFQZWNDAJ5BMGDTNMLF5N'
    }

    try:
        token = get_request_token(AC)
        if token:
            login_info = login_and_host(token)
            if login_info and 'access_token' in login_info:
                return login_info['access_token']
            else:
                logger.error("Failed to retrieve access token from login_info")
                return None
        else:
            logger.error("Failed to generate request token")
            return None
    except Exception as e:
        logger.error(f"Error in authentication: {e}")
        return None

def fetch_historical_data(instrument_token, start_date, end_date, headers):
    """
    Fetch daily historical data from start_date to end_date for a given instrument_token,
    using the provided request headers for authentication.
    """

    all_data = []
    current_start = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    retry_count = 0
    max_retries = 3

    while current_start <= end_date_obj:
        # Calculate up to 1 year from current_start or end_date, whichever is earlier
        next_end = current_start + timedelta(days=365)
        current_end = min(next_end, end_date_obj)

        # Strings for the request
        from_str = current_start.strftime('%Y-%m-%d')
        to_str = current_end.strftime('%Y-%m-%d')

        params = {
            "from": from_str,
            "to": to_str
        }

        try:
            logging.info(f"Fetching data for token {instrument_token} from {from_str} to {to_str}")
            response = requests.get(
                f"{KITE_API_URL}/{instrument_token}/day", 
                headers=headers, 
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'candles' in data['data']:
                    all_data.extend(data['data']['candles'])
                    retry_count = 0  # Reset retry on success
                else:
                    logging.warning(f"Unexpected response structure for {instrument_token}: {data}")
                    retry_count += 1
            else:
                logging.error(f"Failed to fetch data for {instrument_token}: {response.text}")
                retry_count += 1
            
            # Move current_start one day beyond current_end
            current_start = current_end + timedelta(days=1)
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
    Upsert data for a single instrument using SQLAlchemy.
    Table: tblresearch (indices, nav, date, direction, net_change, updated_at)
    The conflict resolution is on (indices, date).
    """
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

def process_daily_data():
    """Process and save data for the current day."""
    try:
        current_date = datetime.now().date()
        logger.info(f"Starting data fetch process for {current_date}")

        # Get authentication token
        access_token = get_auth_token()
        if not access_token:
            logger.error("No access token received; aborting daily data fetch.")
            return

        # Set up headers with new token
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token cme7fe5bmnl0wfqa:{access_token}"
        }

        # Create database engine and session
        engine = sqlalchemy.create_engine(DB_CONNECTION_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()

        failed_instruments = []
        error_messages = []
        success_count = 0

        try:
            for token, instrument_name in tqdm(INSTRUMENTS, desc="Fetching data"):
                try:
                    logger.info(f"Processing {instrument_name}")
                    
                    # Fetch historical data for current date only
                    # (If you want full history, change the dates accordingly)
                    date_str = current_date.strftime('%Y-%m-%d')
                    historical_data = fetch_historical_data(
                        token,
                        date_str,
                        date_str,
                        headers
                    )
                    
                    if not historical_data.empty:
                        if upsert_instrument_data(session, instrument_name, historical_data):
                            success_count += 1
                        else:
                            failed_instruments.append(instrument_name)
                            error_messages.append(f"Failed to upsert data for {instrument_name}")
                    else:
                        failed_instruments.append(instrument_name)
                        error_messages.append(f"No data received for {instrument_name}")
                    
                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    error_msg = f"Error processing {instrument_name}: {str(e)}"
                    logger.error(error_msg)
                    failed_instruments.append(instrument_name)
                    error_messages.append(error_msg)

        finally:
            session.close()

        if failed_instruments:
            logger.error("Failed instruments: " + ", ".join(failed_instruments))
            for msg in error_messages:
                logger.error(msg)
        else:
            logger.info("Daily data fetch completed successfully")

    except Exception as e:
        error_msg = f"Critical error in daily process: {str(e)}"
        logger.error(error_msg)

def get_request_token(AC):
    """
    Use Selenium to log in to Kite and retrieve the request_token.
    This function automates TOTP entry via pyotp, 
    so no human intervention is required if everything is correct.
    """
    driver = None
    try:
        url_ = f"https://kite.trade/connect/login?api_key={AC['api_key']}&v=3"
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), 
            options=options
        )
        
        logger.info("Headless Chrome Initialized")
        driver.get(url_)
        time.sleep(random.choice([5, 6, 7]))
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div#container input[type="text"]'))
        )
        
        driver.find_element(By.CSS_SELECTOR, "div#container input[type='text']").send_keys(AC['USER_ID'])
        logger.info("Login ID Submitted")
        time.sleep(random.choice([2, 3, 4]))
        
        driver.find_element(By.CSS_SELECTOR, "div#container input[type='password']").send_keys(AC['PASS'])
        logger.info("Login Password Submitted")
        time.sleep(random.choice([2, 3, 4]))
        
        driver.find_element(By.CSS_SELECTOR, "div#container button[type='submit']").click()
        logger.info("Login Submit Clicked")
        time.sleep(random.choice([5, 6, 7]))

        # Generate TOTP
        totp = pyotp.TOTP(AC['pin'])
        current_pin = totp.now()
        
        # Wait for next TOTP to rotate if desired
        # (Alternatively, you can just call totp.now() once,
        #  but you appear to be waiting for the next code in the snippet.)
        while True:
            new_pin = totp.now()
            logger.info(f'Generated TOTP: {new_pin}')
            if new_pin != current_pin:
                break
            time.sleep(2)

        # Fill TOTP
        driver.find_element(By.XPATH, "//input[@label='External TOTP']").send_keys(new_pin)
        logger.info("TOTP Submitted")
        time.sleep(random.choice([1, 2, 3]))

        # Check the URL for request_token
        URL = driver.current_url
        logger.info(f"Final URL after login: {URL}")
        parsed_url = urlparse(URL)
        url_elements = parse_qs(parsed_url.query)

        if 'request_token' in url_elements:
            request_token = url_elements['request_token'][0]
            logger.info("Request token generated successfully")
        else:
            logger.error("Could not parse request_token from URL.")
            request_token = None

        driver.quit()
        return request_token
    
    except Exception as e:
        logger.error(f'Error in get_request_token: {e}')
        if driver:
            driver.quit()
        return None

def login_and_host(request_token):
    """
    Generates a Kite session, sets the access token, 
    stores environment variables, and saves instrument data locally.
    """
    AC = {
        'api_key': os.environ.get('KITE_API_KEY'),
        'api_secret': os.environ.get('KITE_API_SECRET')
    }
    
    api_key = AC['api_key']
    api_secret = AC['api_secret']
    if not all([api_key, api_secret]):
        logger.error("API key or secret not found in environment variables.")
        return None

    logger.info("Generating Kite Session")
    kite = KiteConnect(api_key=api_key)
    tokens = kite.generate_session(request_token, api_secret=api_secret)
    logger.info('Kite session generated')

    access_token = tokens["access_token"]
    public_token = tokens["public_token"]
    user_id = tokens["user_id"]

    kite.set_access_token(access_token)
    logger.info('Kite access token set')

    auth = f"&api_key={api_key}&access_token={access_token}"
    logger.info('All tokens generated')
    logger.info(f'Zerodha -- Logged in Successfully at {time.strftime("%d-%b-%Y %A %H:%M:%S", time.localtime())}')

    kws = KiteTicker(api_key, access_token)
    
    login_credentials = {
        'kws': kws,
        'kite': kite,
        'access_token': access_token,
        'public_token': public_token,
        'user_id': user_id,
        'auth': auth,
        'api_key': api_key,
        'api_secret': api_secret,
        'update_time': datetime.now()
    }
    logger.info('Login credentials generated successfully')

    # Fetch all instrument data
    logger.info('Fetching entire instrument list from Kite')
    instrument_id = kite.instruments()
    inst = pd.DataFrame(instrument_id)

    # Save instruments to a gzip file (date-based)
    os.makedirs(INST_PATH, exist_ok=True)
    file_path = f"{INST_PATH}{datetime.now().date().strftime('%Y%m%d')}_inst.pkl.gz"
    try:
        with gzip.open(file_path, 'wb') as file:
            pickle.dump(inst, file)
        logger.info(f"Instrument data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving instrument data: {e}")

    # Update Next.js .env file with the new access token
    nextjs_env_path = './.env'
    update_env_file('KITE_ACCESS_TOKEN', access_token, env_path=nextjs_env_path)
    logger.info('Access token stored in Next.js .env file')
    
    return login_credentials

def run_schedule():
    """Run the scheduler: daily job at 19:00 (7 PM)."""
    schedule.every().day.at("19:00").do(process_daily_data)
    
    logger.info("Scheduler started - waiting for 7 PM daily.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    logger.info("Starting automated data fetch service")
    
    # If you want to run immediately (e.g. for testing), call with "--run-now"
    if len(sys.argv) > 1 and sys.argv[1] == "--run-now":
        process_daily_data()
    else:
        run_schedule()
