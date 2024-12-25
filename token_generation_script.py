import os
import sys
import time
import random
import dotenv
import pyotp
import urllib.parse
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import pandas as pd
import gzip
import pickle
import datetime
from kiteconnect import KiteConnect, KiteTicker
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('./.env')

# Define INST_PATH
INST_PATH = os.getenv('INST_PATH', './data/')
os.makedirs(INST_PATH, exist_ok=True)

def update_env_file(key, value, env_path='.env.local'):
    dotenv.load_dotenv(dotenv_path=env_path)
    env_vars = dotenv.dotenv_values(env_path)
    env_vars[key] = value

    with open(env_path, 'w') as env_file:
        for k, v in env_vars.items():
            env_file.write(f"{k}={v}\n")
    logger.info(f"Updated {key} in {env_path}")

def login_and_host(request_token):
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
    kite.set_access_token(tokens["access_token"])
    logger.info('Initializing Kite access tokens')
    
    access_token = tokens["access_token"]
    public_token = tokens["public_token"]
    user_id = tokens["user_id"]
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
        'update_time': datetime.datetime.now()
    }
    logger.info('Login credentials Hosted Successfully')
    
    logger.info('Fetching all instrument list')
    instrument_id = kite.instruments()
    inst = pd.DataFrame(instrument_id)

    file_path = f"{INST_PATH}{datetime.datetime.now().date().strftime('%Y%m%d')}_inst.pkl.gz"
    with gzip.open(file_path, 'wb') as file:
        pickle.dump(inst, file)
    logger.info(f"Instrument data saved to {file_path}")

    nextjs_env_path = './.env'
    update_env_file('KITE_ACCESS_TOKEN', access_token, env_path=nextjs_env_path)
    logger.info('Access token stored in Next.js .env file')
    
    return login_credentials  # Return the entire login credentials dictionary
def get_request_token(AC):
    try:
        url_ = "https://kite.trade/connect/login?api_key=" + AC['api_key'] + "&v=3"
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        logger.info("Headless Chrome Initialized")
        driver.get(url_)
        time.sleep(random.choice([5, 6, 7]))
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div#container input[type="text"]')))
        
        driver.find_element(By.CSS_SELECTOR, "div#container input[type='text']").send_keys(AC['USER_ID'])
        logger.info("Login ID Submitted")
        time.sleep(random.choice([2, 3, 4]))
        
        driver.find_element(By.CSS_SELECTOR, "div#container input[type='password']").send_keys(AC['PASS'])
        logger.info("Login Pass Submitted")
        time.sleep(random.choice([2, 3, 4]))
        
        driver.find_element(By.CSS_SELECTOR, "div#container button[type='submit']").click()
        logger.info("Login Submit Clicked")
        time.sleep(random.choice([5, 6, 7]))

        totp = pyotp.TOTP(AC['pin'])
        print(totp)
        current_pin = totp.now()
        while True:
            new_pin = totp.now()
            print('generating new pin',new_pin)
            if new_pin != current_pin:
                break
            time.sleep(2)

        driver.find_element(By.XPATH, "//input[@label='External TOTP']").send_keys(new_pin)
        logger.info("TOTP Submitted")
        time.sleep(random.sample([1, 2, 3], 1)[0])

        URL = driver.current_url
        print('URL',URL)
        parsed_url = urlparse(URL)
        url_elements = parse_qs(parsed_url.query)
        request_token = url_elements['request_token'][0]
        driver.quit()
        
        logger.info("Request token generated Successfully")
        return request_token
    
    except Exception as e:
        logger.error(f'Error in get_request_token: {e}')
        driver.quit()
        return None

if __name__ == '__main__':
    AC = {
        'api_key': 'cme7fe5bmnl0wfqa',
        'api_secret': 'no15x3wpugb63yt6ju1jcuzmjo0zp5vc',
        'USER_ID': 'UIK957',
        'PASS': 'Qode@pms123',
        'pin': 'UWXLCMWDU46UFQZWNDAJ5BMGDTNMLF5N'
    }

    token = get_request_token(AC)
    if token:
        login_info = login_and_host(token)
        # Directly print the access token
        print(login_info['access_token'])
    else:
        print("Token generation failed", file=sys.stderr)
