import io
from operator import eq
import os
import shutil
import traceback
from flask import Flask, request, jsonify,make_response, send_file, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from pyxirr import xirr
from datetime import datetime
import pyxirr
import logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
ALLOWED_ORIGINS = [
    "https://qodeinvest.com",
    "https://www.qodeinvest.com",
    "https://qodepreview.netlify.app",
    "https://www.qodepreview.netlify.app",
    "http://localhost:5173"
]

CORS(app, 
     resources={
        r"/calculate_portfolio": {"origins": ALLOWED_ORIGINS},
        r"/upload_excel": {"origins": ALLOWED_ORIGINS},
        r"/revert_changes": {"origins": ALLOWED_ORIGINS},
        r"/get_date_range": {"origins": ALLOWED_ORIGINS}
     },
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

original_df_path = r'C:\Users\tech\Desktop\development\python\portfolio-visualiser\csv\All weather portfolio.csv'
backup_df_path = r'C:\Users\tech\Desktop\development\python\portfolio-visualiser\csv\Backup_All weather portfolio.csv'
temp_merged_df_path = r'C:\Users\tech\Desktop\development\python\portfolio-visualiser\csv\Temp_Merged_All weather portfolio.csv'

global global_equity_curve_data


# Make a backup of the original CSV if it doesn't exist
if not os.path.exists(backup_df_path):
    shutil.copyfile(original_df_path, backup_df_path)

@app.route('/upload_excel', methods=['POST', 'OPTIONS'])
def upload_excel():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "https://qodeinvest.com")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        response.status_code = 200
        return response

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']

    # Read the existing CSV file with date parsing
    existing_df = pd.read_csv(original_df_path, parse_dates=['Date'], dayfirst=True).set_index('Date')

    # Read the uploaded CSV file
    uploaded_df = pd.read_csv(file)

    # Parse dates separately with dayfirst=True
    if 'Date' in uploaded_df.columns:
        uploaded_df['Date'] = pd.to_datetime(uploaded_df['Date'], dayfirst=True)

    # Merge the dataframes
    merged_df = add_csv(existing_df.reset_index(), uploaded_df)

    # Ensure 'Date' is in the 'dd-mm-yyyy' format
    merged_df['Date'] = merged_df['Date'].dt.strftime('%d-%m-%Y')

    # Save the merged DataFrame to a temporary file
    merged_df.to_csv(temp_merged_df_path, index=False)

    columns = [col for col in uploaded_df.columns.tolist() if col != 'Date']

    # Return the column names in the response
    return jsonify({
        "message": "CSV file uploaded and merged successfully.",
        "columns": columns
    }), 200

@app.route('/get_date_range', methods=['GET'])
def get_date_range():
    try:
        # Load the dataframe based on the available path
        data_path = temp_merged_df_path if os.path.exists(temp_merged_df_path) else original_df_path
        df = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
        df.set_index('Date', inplace=True)

        # Get the min and max dates
        min_date = df.index.min()
        max_date = df.index.max()

        # Format the dates as 'dd-mm-yyyy'
        min_date_str = min_date.strftime('%d-%m-%Y')
        max_date_str = max_date.strftime('%d-%m-%Y')

        return jsonify({
            "min_date": min_date_str,
            "max_date": max_date_str
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/download-excel', methods=['GET'])
def download_excel():
    try:
        # Load the dataframe based on the available path (use temp_merged_df_path or original_df_path)
        data_path = temp_merged_df_path if os.path.exists(temp_merged_df_path) else original_df_path
        df = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
        
        # Process or filter the dataframe if needed (e.g., perform calculations)
        # For now, we're just downloading the loaded dataframe
        
        # Save the dataframe to an Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Portfolio Data')

        output.seek(0)  # Go to the beginning of the stream
        
        # Send the file as a response
        return send_file(
            output,
            as_attachment=True,
            download_name="portfolio_data.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/revert_changes', methods=['POST'])
def revert_changes():
    try:
        # Remove the temporary merged file if it exists
        if os.path.exists(temp_merged_df_path):
            os.remove(temp_merged_df_path)
        
        return jsonify({"message": "Changes reverted successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def add_csv(df_initial1, df_added1):
    df_initial = df_initial1.copy()
    df_added = df_added1.copy()

    # Handle data types and clean any formatting
    for column in df_added.columns:
        if df_added[column].dtype == 'object':
            df_added[column] = df_added[column].replace({',': ''}, regex=True)
            try:
                df_added[column] = df_added[column].astype(float)
            except ValueError:
                pass  # Handle or log error if conversion fails

    # Convert 'Date' to datetime
    df_initial['Date'] = pd.to_datetime(df_initial['Date'])
    df_added['Date'] = pd.to_datetime(df_added['Date'])

    # Drop duplicate dates in the initial DataFrame
    df_initial.drop_duplicates(subset=['Date'], inplace=True)

    # Ensure no duplicates in the Date index before merging
    df_added = df_added.drop_duplicates(subset=['Date'])

    # Merge the DataFrames
    merged_df = pd.merge(df_initial, df_added, on='Date', how='outer', suffixes=(False, False))

    # Sort by date after merging
    merged_df.sort_values(by='Date', inplace=True)

    return merged_df

@app.route('/calculate_portfolio', methods=['GET', 'POST', 'OPTIONS'])
def calculate_portfolio(dataframe=None):
    if request.method == 'OPTIONS':
        response = make_response()
        origin = request.headers.get('Origin')

        allowed_origins = [
            "https://qodeinvest.com",
            "https://qodepreview.netlify.app",
            "https://www.qodepreview.netlify.app",
            "http://localhost:5173"
            
        ]

        if origin in allowed_origins:
            response.headers.add("Access-Control-Allow-Origin", origin)

        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")

        response.status_code = 200
        return response

    try:
        data_path = temp_merged_df_path if os.path.exists(temp_merged_df_path) else original_df_path
        df = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True, on_bad_lines='skip') if dataframe is None else dataframe
        
        # Parse and validate the 'Date' column
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

        df = df.dropna(subset=['Date'])

        if df.empty:
            logging.error("No valid dates found in the provided data.")
            return jsonify({"error": "No valid dates found in the provided data."}), 400

        df.set_index('Date', inplace=True)

        min_date = df.index.min()
        max_date = df.index.max()

        if pd.isnull(min_date) or pd.isnull(max_date):
            logging.error("The date range contains invalid dates.")
            return jsonify({"error": "The date range contains invalid dates."}), 400

        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        df = df.reindex(full_date_range).ffill()
        logging.info(f"Date column types after parsing: {df.index.dtype}")

        # Log the incoming data for debugging
        data = request.json
        logging.info(f"Incoming JSON data: {data}")

        if not data:
            logging.error("No data provided in the request.")
            return jsonify({"error": "No data provided in the request."}), 400

        # Validate the required fields
        required_fields = ['invest_amount', 'cash_percent', 'start_date', 'end_date', 'selected_systems', 'frequency']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract data and parse dates
        invest_amount = data['invest_amount']
        cash_percent = data['cash_percent'] / 100
        start_date = pd.to_datetime(data['start_date'], format='%d-%m-%Y', errors='coerce')
        end_date = pd.to_datetime(data['end_date'], format='%d-%m-%Y', errors='coerce')

        if pd.isnull(start_date) or pd.isnull(end_date):
            logging.error("Invalid start or end date provided.")
            return jsonify({"error": "Invalid start or end date provided."}), 400
        if start_date > end_date:
            logging.error("Start date cannot be after end date.")
            return jsonify({"error": "Start date cannot be after end date."}), 400

        selected_systems_list = data['selected_systems']
        selected_systems_names = extract_system_names(selected_systems_list)
        selected_debtfund_list = data.get('selected_debtfunds', [])
        selected_debtfund_names = extract_debtfund_names(selected_debtfund_list)
        frequency = data['frequency']

        logging.info(f"Processed input data: invest_amount={invest_amount}, cash_percent={cash_percent}, start_date={start_date}, end_date={end_date}")
        logging.info(f"Selected systems: {selected_systems_names}, Selected debt funds: {selected_debtfund_names}")

        weights_systems = calculate_weights(selected_systems_list)
        leverage_systems = calculate_leverage_values(selected_systems_list)
        freq = get_frequency_value(frequency)

        new_DF = required_df(df, start_date, end_date, selected_systems_names)
        print(f'New DF: {new_DF}')
        df_multiplied = percentage_df_fun(new_DF, leverage_systems)

        new_inv_amount, cash = input_cash(invest_amount, cash_percent)
        initial_df_cash = initialize_cash_df(new_DF, cash)

        if frequency != 'no':
            result = weights_rebalance_frequency(
                new_DF, weights_systems, new_inv_amount, df_multiplied, freq, cash_percent, initial_df_cash, invest_amount
            )
        else:
            result = weights_rebalancing_main(
                new_DF, weights_systems, new_inv_amount, df_multiplied, initial_df_cash
            )

        if isinstance(result, tuple):
            df_sum = result[0]
        else:
            df_sum = result

        if selected_debtfund_names:
            logging.info("=== DEBT FUND PROCESSING START ===")
            logging.info(f"Selected debt funds: {selected_debtfund_names}")
            
            df_debts = required_df(df, start_date, end_date, selected_debtfund_names)
            logging.info(f"Debt funds dataframe:\n{df_debts.head()}")
            logging.info(f"Debt funds dataframe shape: {df_debts.shape}")
            
            debt_leverage_values = calculate_leverage_values(selected_debtfund_list)
            weights_debts = calculate_weights(selected_debtfund_list)
            
            logging.info(f"Debt leverage values: {debt_leverage_values}")
            logging.info(f"Debt weights: {weights_debts}")
            
            df_multiplied_debt = percentage_df_fun(df_debts, debt_leverage_values)
            logging.info(f"Multiplied debt dataframe:\n{df_multiplied_debt.head()}")
            logging.info(f"Multiplied debt dataframe shape: {df_multiplied_debt.shape}")
            
            df_sum_debt = weights_rebalance_frequency_debt(
                df_debts, weights_debts, df_multiplied_debt, freq, new_inv_amount
            )
            logging.info(f"Final debt sum dataframe:\n{df_sum_debt.head()}")
            logging.info(f"Final debt sum dataframe shape: {df_sum_debt.shape}")
            
            df_sum = add_debtfund(df_sum_debt, df_sum)
            logging.info(f"Combined dataframe after adding debt funds:\n{df_sum.head()}")
            logging.info(f"Combined dataframe shape: {df_sum.shape}")
            logging.info("=== DEBT FUND PROCESSING END ===")

        validate_selected_systems(selected_systems_list)

        if 'Date' not in df_sum.columns:
            df_sum = df_sum.reset_index().rename(columns={'index': 'Date'})

        # Log the analysis data before returning it
        logging.info(f"Final analysis dataframe:\n{df_sum.head()}")

        try:
            result = show_analysis(df_sum)
            # No date conversion here
            global global_equity_curve_data
            global_equity_curve_data = result['equity_curve_data']
            print(global_equity_curve_data)
            download_link = url_for('download_equity_curve')
            logging.info('show_analysis completed successfully')
            return jsonify({'result': result, 'download_link': download_link})
        except Exception as analysis_error:
            logging.error(f"Error in show_analysis: {str(analysis_error)}")
            logging.error(traceback.format_exc())
            return jsonify({"error": f"Error in analysis: {str(analysis_error)}", "traceback": traceback.format_exc()}), 500



    except Exception as e:
        logging.error(f"Error in calculate_portfolio: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/download_equity_curve', methods=['GET'])
def download_equity_curve():
    # Access the equity_curve_data from the global variable
    global global_equity_curve_data

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(global_equity_curve_data)

    # Convert the 'Date' column to datetime format with dayfirst=True
    if 'Date' in df.columns:
        # Use dayfirst=True to correctly parse dates like '8/7/2013' as July 8, 2013
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        # Optionally, format the dates as strings in a consistent format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Create an in-memory Excel file
    excel_file = io.BytesIO()
    df.to_excel(excel_file, index=False, sheet_name='Equity Curve')
    excel_file.seek(0)

    # Return the Excel file for download
    return send_file(
        excel_file,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='equity_curve.xlsx'
    )

def get_frequency_value(frequency):
    """Converts frequency input into numeric value for rebalancing calculations."""
    if frequency == 'daily':
        return 1
    elif frequency == 'weekly':
        return 7
    elif frequency == 'monthly':
        return 30
    elif frequency == 'yearly':
        return 365
    return 1  # Default to daily if unspecified

def convert_nan_to_none(data):
    """Recursively convert NaN, NaT values to None and datetime to strings in a nested dictionary or list."""
    if isinstance(data, dict):
        return {key: convert_nan_to_none(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_none(item) for item in data]
    elif isinstance(data, pd.Series):
        # Convert Series to a list with converted values
        return [convert_nan_to_none(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to a dictionary of lists
        return data.applymap(convert_nan_to_none).to_dict(orient='list')
    elif isinstance(data, (float, np.float64)) and np.isnan(data):
        return None
    elif isinstance(data, pd.Timestamp):
        if pd.isna(data):
            return None  # Convert NaT to None directly
        else:
            return data.strftime('%Y-%m-%d')  # Format datetime objects
    elif pd.api.types.is_scalar(data) and pd.isna(data):
        return None
    else:
        return data

def input_cash(invest_amount, cash_percent): #function to take cash as input

    

    cash = invest_amount*cash_percent

    new_inv_amount = invest_amount - cash

    return new_inv_amount, cash

def required_df(df, start_date, end_date, systems):
    
    cols = list(df.columns)
    cols = cols[1:]


    df1 = df[systems]


    if (start_date!='0' and end_date!='0'):
        df1 = df1.loc[start_date:end_date]

    start_index = df1.first_valid_index()

    end_index = df1.last_valid_index()

    df2 = df1.loc[start_index:end_index]


    return df2

def statsbox(df1): #function to print the stats
    
    df_sum = df1.copy()
    print(f'DF SUM: {df_sum}')
    car = calculate_car(df_sum) # getting car value

    ##print(f'CAR: {car}')

    max_dd, avg_dd, drawdown_table, top_10_worst_drawdowns, max_peaktopeak, drawdown_periods = cal_maxdrawdown(df_sum)  # getting drawdown analysis 


    ##print(f'Max DD: {max_dd}')

    ##print(f'Average DD: {avg_dd}')

    carbymdd = cardivmdd(car, max_dd)
    ##print(f'CAR/MDD: {carbymdd}')


    max_gain, max_loss = daily_loss_gain(df_sum)

    ##print(f'Max daily gain: {max_gain}')
    ##print(f'Max daily loss: {max_loss}')

    ##print(top_10_worst_drawdowns) # printing top 10 worst drawdowns

    car = round(car, 2)
    max_dd = round(max_dd, 2)
    avg_dd = round(avg_dd, 2)
    carbymdd = round(carbymdd, 2)
    max_gain = round(max_gain, 2)
    max_loss = round(max_loss, 2)

    return car, max_dd, avg_dd, carbymdd, max_gain, max_loss, top_10_worst_drawdowns, drawdown_table, max_peaktopeak

def calculate_car(df1):  # function to calculate CAR
    print('calculate_car', df1)
    df = df1.copy()

    # Check if 'Date' column exists and handle it appropriately
    if 'Date' not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    
    # Convert 'Date' to datetime without setting it as index
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Check for duplicates in 'Date' and remove or aggregate them if necessary
    if df['Date'].duplicated().any():
        df = df.drop_duplicates(subset='Date', keep='last')
    
    # Sort the dataframe by date to ensure correct order
    df = df.sort_values('Date')
    print('dffffffffffff', df)
    initial_index = df['Portfolio Value'].first_valid_index()
    print('initial_index',initial_index)
    initial_value = -df['Portfolio Value'].iloc[initial_index]
    final_value = df['Portfolio Value'].iloc[-1]

    print(f'Dataframe: {df}')
    print(f'Initial value: {initial_value}')
    print(f'Final Value: {final_value}')

    # Use the first and last date for XIRR calculation
    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]

    car2 = xirr([start_date, end_date], [initial_value, final_value])  # using xirr function
    car2 = car2 * 100

    return car2

def validate_data(df):
    assert df['Date'].dtype == 'datetime64[ns]', "Date column should be datetime type"
    assert df['Portfolio Value'].dtype == 'float64', "Portfolio Value should be float type"
    logging.info("Data validation passed successfully")

def cardivmdd(car, mdd): 
    return car / abs(mdd)

def convert_to_datetime(date_str):
    # If the input is already a Timestamp, return it as is
    if isinstance(date_str, pd.Timestamp):
        return date_str
    
    # If the input is a string, try converting it to datetime
    elif isinstance(date_str, str):
        try:
            return datetime.strptime(date_str, '%d-%m-%Y')
        except ValueError:
            # Handle the case where the date string is not in the expected format
            return None

    # If it's not a string or Timestamp, return None
    return None

def cal_maxdrawdown(df1):
    df = df1.copy()

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Drop any rows where the 'Date' conversion failed (optional)
    df = df.dropna(subset=['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Sort the DataFrame by the index
    df = df.sort_index()

    # Calculate previous peaks
    df['Previous Peaks'] = df['Portfolio Value'].cummax()

    # Calculate drawdown as a percentage
    drawdown = (df['Portfolio Value'] / df['Previous Peaks'] - 1) * 100

    # Round drawdown values
    max_drawdown = round(drawdown.min(), 2)  # Minimum drawdown value
    average_drawdown = round(drawdown[drawdown < 0].mean(), 2)  # Average of negative drawdowns

    # Initialize drawdown tracking variables
    drawdown_periods = []
    in_drawdown = False
    peak_date = None
    peak_value = None
    trough_date = None
    trough_value = 0

    # Helper function to format dates
    def format_date(date):
        return date.strftime('%d-%m-%Y') if pd.notna(date) else 'N/A'

    # Helper function to get a single value from a potential Series
    def get_single_value(value):
        return value.iloc[0] if isinstance(value, pd.Series) else value

    # Iterate over the DataFrame rows to detect peaks, troughs, and recoveries
    for date, row in df.iterrows():
        value = get_single_value(row['Portfolio Value'])
        if not in_drawdown:
            if peak_value is not None and value < peak_value:  # Start of a drawdown
                in_drawdown = True
                trough_date = date
                trough_value = get_single_value(drawdown.loc[date])
        elif in_drawdown:
            current_trough_value = get_single_value(df.loc[trough_date, 'Portfolio Value'])
            if value < current_trough_value:  # New trough within the drawdown
                trough_date = date
                trough_value = get_single_value(drawdown.loc[date])
            elif value >= peak_value:  # Recovery detected
                recovery_date = date
                drawdown_periods.append({
                    'Peak_Date': format_date(peak_date),
                    'Drawdown_Date': format_date(trough_date),
                    'Recovery_Date': format_date(recovery_date),
                    'Drawdown': round(trough_value, 2),
                    'Days': (recovery_date - trough_date).days if pd.notna(trough_date) and pd.notna(recovery_date) else None
                })
                in_drawdown = False
                peak_date = date
                peak_value = value
        # Update peak value when not in a drawdown
        if not in_drawdown:
            if peak_value is None or value > peak_value:
                peak_date = date
                peak_value = value

    # If still in a drawdown, record it as ongoing
    if in_drawdown:
        drawdown_periods.append({
            'Peak_Date': format_date(peak_date),
            'Drawdown_Date': format_date(trough_date),
            'Recovery_Date': 'Ongoing',
            'Drawdown': round(trough_value, 2),
            'Days': None  # Set to None for ongoing drawdowns
        })

    # Convert drawdown periods to DataFrame
    df_drawdown = pd.DataFrame(drawdown_periods)

    # Correctly calculate the "Days between Drawdown and Recovery Date"
    df_drawdown['Days between Drawdown and Recovery Date'] = df_drawdown.apply(
        lambda row: (pd.to_datetime(row['Recovery_Date'], format='%d-%m-%Y') - pd.to_datetime(row['Drawdown_Date'], format='%d-%m-%Y')).days 
        if row['Recovery_Date'] != 'Ongoing' else pd.NA, axis=1)  # Set to pd.NA for ongoing drawdowns

    # Sort by drawdown percentage to find the worst periods
    try:
        df_drawdown_sorted = df_drawdown.sort_values('Drawdown', ascending=True)
    except Exception as e:
        print(f"Error sorting drawdowns: {e}")
        print("Debug: df_drawdown datatypes")
        print(df_drawdown.dtypes)
        df_drawdown_sorted = df_drawdown  # Use unsorted dataframe if sorting fails

    # Extract the top 10 worst drawdowns and round the values
    top_10_worst_drawdowns = df_drawdown_sorted.head(10).copy()
    top_10_worst_drawdowns['Drawdown'] = top_10_worst_drawdowns['Drawdown'].round(2)

    # Create a DataFrame of all drawdown values and round them
    drawdown_df = drawdown.round(2).to_frame(name='Drawdown')

    # Calculate peak-to-peak days and find the longest ones
    peak2peak_days = df_drawdown['Days'].dropna().sort_values(ascending=False)[:10]
    max_peak_to_peak = pd.DataFrame({'Max peak to peak': peak2peak_days})

    # Return all calculated metrics, including the updated df_drawdown with "Days between Drawdown and Recovery Date"
    return max_drawdown, average_drawdown, drawdown_df, top_10_worst_drawdowns, max_peak_to_peak, df_drawdown

def calculate_recovery_dates(df):
    recovery_dates = []
    for i, row in df.iterrows():
        future_values = df.loc[i:, 'Portfolio Value']
        recovery_index = future_values[future_values > row['Previous Peaks']].index
        if not recovery_index.empty:
            recovery_dates.append(recovery_index[0])
        else:
            recovery_dates.append('Ongoing')
    return recovery_dates

def daily_loss_gain(df1):
    print('daily_loss_gain', df1)
    df = df1.copy()

    # Ensure 'Date' is set as the index
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    
    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Backfill then forward fill NaN values
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    # Replace zero values with a small number to avoid division by zero
    df = df.replace(0, 1e-10)

    # Calculate percentage change without using fill_method
    returns = df.pct_change()

    # Replace infinity values with NaN
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop NaN values
    returns.dropna(inplace=True)

    # Find max gain and max loss in percentage
    max_gain = returns.max() * 100
    max_loss = returns.min() * 100
    
    print("Returns:", returns)
    print("Max gain:", max_gain)
    print("Max loss:", max_loss)

    # Check if 'Portfolio Value' column exists, otherwise use the first column
    if 'Portfolio Value' in max_gain.index:
        return max_gain['Portfolio Value'], max_loss['Portfolio Value']
    else:
        return max_gain.iloc[0], max_loss.iloc[0]

def monthly_pl_table(df1): # function to calculate monthly PL table

    df = df1.copy()
    print("monthly_pl_table",df)
    # First, ensure the 'Date' column is of string type (in case it's not)
    df['Date'] = df['Date'].astype(str)

    # Convert the 'Date' column to datetime, coercing any invalid entries to NaT
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Check for any NaT values (which means invalid dates were found)
    invalid_dates = df[df['Date'].isna()]
    if not invalid_dates.empty:
        print("Invalid dates found:", invalid_dates)

    # Now, you can safely set the 'Date' column as the index if needed
    df.set_index('Date', inplace=True)  


    # Calculate the monthly percentage change
    monthly_portfolio_value = df['Portfolio Value'].resample('ME').agg(['first', 'last'])

    # Calculate the percentage change for each month
    monthly_percentage_change = (monthly_portfolio_value['last'] / monthly_portfolio_value['last'].shift(1) - 1)

    yearly_portfolio_value = df['Portfolio Value'].resample('YE').agg(['first', 'last'])

    yearly_percentage_change = (yearly_portfolio_value['last'] / yearly_portfolio_value['first'] - 1)

    yearly_percentage_change.iloc[0] = yearly_portfolio_value.iloc[0,1] / yearly_portfolio_value.iloc[0,0] - 1
    # For the first month of each year, use the first value of that month

    monthly_percentage_change.iloc[0] = monthly_portfolio_value.iloc[0,1] / monthly_portfolio_value.iloc[0,0] - 1

    # Convert the index to a DataFrame to extract year and month information
    monthly_percentage_change_df = pd.DataFrame(monthly_percentage_change * 100)
    monthly_percentage_change_df['Year'] = monthly_percentage_change_df.index.year
    monthly_percentage_change_df['Month'] = monthly_percentage_change_df.index.month_name()

    # Pivot the DataFrame to get the desired tabular format
    pivot_table = pd.pivot_table(monthly_percentage_change_df, values='last', index='Year', columns='Month', fill_value=None)

    # Rename the columns to have a consistent order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    pivot_table = pivot_table.reindex(columns=month_order)

    pivot_table['Total'] = yearly_percentage_change.values * 100
    
    #pivot_table['Total'] = pivot_table.sum(axis=1)
    # Print or use the resulting table

    return pivot_table

def show_analysis(df_sum_new):
    logging.info('Entering show_analysis function')
    logging.info(f'Input dataframe shape: {df_sum_new.shape}')
    logging.info(f'Input dataframe columns: {df_sum_new.columns.tolist()}')
    logging.info(f'Input dataframe dtypes: {df_sum_new.dtypes}')

    try:
        if isinstance(df_sum_new, tuple):
            df_sum_new = df_sum_new[0]
        
        # Ensure 'Date' column is present and properly formatted
        if 'Date' not in df_sum_new.columns:
            df_sum_new = df_sum_new.reset_index()
        
        if 'Date' not in df_sum_new.columns:
            raise ValueError("The dataframe does not contain a 'Date' column after reset_index.")
        
        # Convert 'Date' column to datetime, assuming it's in 'dd-mm-yyyy' format
        df_sum_new['Date'] = pd.to_datetime(df_sum_new['Date'], format='%d-%m-%Y')
        
        # Format 'Date' column to 'dd-mm-yyyy' for consistency
        df_sum_new['Date'] = df_sum_new['Date'].dt.strftime('%d-%m-%Y')
        
        logging.info('Date column processed successfully')
        
        # Generate the required analysis
        logging.info('About to call statsbox function')
        car, max_dd, avg_dd, carbymdd, max_gain, max_loss, top_10_worst_drawdowns, drawdown_table, max_peaktopeak = statsbox(df_sum_new)
        logging.info('statsbox function completed successfully')
        logging.info('About to call monthly_pl_table function')
        monthly_pl_pivot = monthly_pl_table(df_sum_new)
        logging.info('monthly_pl_table function completed successfully')
        
        logging.info('Analysis calculations completed')
        
        # Format equity curve data
        equity_curve_data = df_sum_new.apply(lambda row: {
            'Date': row['Date'],
            'NAV': row['Portfolio Value']
        }, axis=1).tolist()
        
        # Format drawdown data
        drawdown_data = drawdown_table.reset_index().apply(lambda row: {
            'Date': row['Date'].strftime('%d-%m-%Y') if pd.notnull(row['Date']) else None,
            'Drawdown': row['Drawdown'] if pd.notnull(row['Drawdown']) else None
        }, axis=1).tolist()
        
        # Prepare peak to peak data
        max_peaktopeak.reset_index(drop=True, inplace=True)
        max_peaktopeak.index = max_peaktopeak.index + 1
        max_peaktopeak['Rank'] = max_peaktopeak.index
        peak_to_peak_data = max_peaktopeak[['Rank', 'Max peak to peak']].to_dict('records')
        
        # Calculate CAGR
        cagrData = calculate_xirr(df_sum_new)
        
        # Prepare monthly performance data
        rounded_pivot_table = monthly_pl_pivot.round(2)
        monthly_pl_table_data = rounded_pivot_table.reset_index().to_dict('records')
        
        logging.info('Data formatting completed')
        
        # Prepare response
        response = {
            'equity_curve_data': equity_curve_data,
            'drawdown_data': drawdown_data,
            'peak_to_peak_data': peak_to_peak_data,
            'monthly_pl_table': monthly_pl_table_data,
            'car': car,
            'cagrData': cagrData,
            'max_dd': max_dd,
            'avg_dd': avg_dd,
            'carbymdd': carbymdd,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'top_10_worst_drawdowns': top_10_worst_drawdowns.to_dict('records'),
        }
        
        # Convert NaNs to None for JSON serialization
        response = convert_nan_to_none(response)
        
        logging.info('Response prepared successfully')
        return (response)
    
    except Exception as e:
        logging.error(f"Error in show_analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def calculate_xirr(df1):
    df = df1.copy()

    # First, ensure the 'Date' column is of string type (in case it's not)
    df['Date'] = df['Date'].astype(str)

    # Convert the 'Date' column to datetime, coercing any invalid entries to NaT
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Check for any NaT values (which means invalid dates were found)
    invalid_dates = df[df['Date'].isna()]
    if not invalid_dates.empty:
        print("Invalid dates found:", invalid_dates)

    # Now, you can safely set the 'Date' column as the index if needed
    df.set_index('Date', inplace=True)

    df = df.sort_index()

    df_resampled = df.resample('D').ffill()
    df_resampled = df_resampled.reset_index().rename(columns={'index': 'Date'})

    new_cols = ['5 yr CAGR', '3 yr CAGR', '1 yr CAGR']
    new_df = pd.DataFrame(index=df_resampled.index, columns=new_cols)

    periods = {'5 yr CAGR': 365 * 5, '3 yr CAGR': 365 * 3, '1 yr CAGR': 365}

    for col, gap in periods.items():
        for d in range(gap, len(df_resampled)):
            initial_value = df_resampled['Portfolio Value'].iloc[d-gap]
            final_value = df_resampled['Portfolio Value'].iloc[d]
            
            if initial_value > 0 and final_value > 0:
                cagr = (final_value / initial_value) ** (365 / gap) - 1
                new_df.at[df_resampled.index[d], col] = round(cagr * 100, 2)
            else:
                new_df.at[df_resampled.index[d], col] = None

    print('calculate_xirr', new_df)

    # Convert the 'Date' column in the df_resampled to DatetimeIndex and format it
    new_df.index = pd.to_datetime(df_resampled['Date']).dt.strftime('%d-%m-%Y')

    new_df = new_df.apply(pd.to_numeric, errors='coerce')

    result_dict = {}
    for column_name in new_cols:
        top_5_values = new_df[column_name].nlargest(5)
        top_5_min_values = new_df[column_name].nsmallest(5)
        average_value = round(new_df[column_name].dropna().mean(), 2)
        
        result_dict[column_name] = {
            'top_5_max_values': top_5_values,
            'top_5_min_values': top_5_min_values,
            'average_value': average_value
        }

    return result_dict

def weights_rebalance_frequency(df, weights, invest_amount, df_multiplied, freq, cash_percent, initial_df_cash, total_invest_amount):
    initial_port_df = df.copy().astype(float)
    cash_amount_end = total_invest_amount * cash_percent
    sorted_total_dates, stock_dates = date_dict(initial_port_df)
    print('initial_port_dfffff',initial_port_df)
    print('sort_dates',sorted_total_dates)
    # Convert the DataFrame index to datetime if not already
    initial_port_df.index = pd.to_datetime(initial_port_df.index, format='%d-%m-%Y')
    initial_df_cash.index = pd.to_datetime(initial_df_cash.index, format='%d-%m-%Y')
    df_multiplied.index = pd.to_datetime(df_multiplied.index, format='%d-%m-%Y')

    print('sorted_total_dates', sorted_total_dates)
    
    for i, date in enumerate(sorted_total_dates):
        # Convert date to datetime for matching
        date = pd.to_datetime(date, format='%d-%m-%Y')
        
        new_weights_bool = bool_list(date, initial_port_df, stock_dates)
        print('new_weights_bool', new_weights_bool)
        print('weights', weights)
        new_weights = rebalance_weights(weights, new_weights_bool)   # Pass the entire weights dictionary
        start_day = date
        
        if i < (len(sorted_total_dates) - 1):
            end_day = pd.to_datetime(sorted_total_dates[i+1], format='%d-%m-%Y')
            end_day_index = initial_port_df.index.get_loc(end_day) - 1
            end_day_new = initial_port_df.index[end_day_index]
        else:
            end_day_new = None
            
        date_index = initial_port_df.index.get_loc(start_day)
        start_index = freq - (date_index % freq)
        period_port_df = initial_port_df.loc[start_day:end_day_new]
        period_leverage_df = df_multiplied.loc[start_day:end_day_new]
        period_df_cash = initial_df_cash.loc[start_day:end_day_new]
        
        rebalanced_portfolio, cash_rebalanced = rebalancing2(
            period_port_df, period_leverage_df, new_weights, freq, invest_amount, 
            start_index, cash_percent, total_invest_amount, period_df_cash, cash_amount_end)
        
        invest_amount = rebalanced_portfolio.iloc[-1].sum()
        cash_amount_end = cash_rebalanced.iloc[-1].item()
        total_invest_amount = invest_amount + cash_amount_end
        initial_port_df.loc[start_day:end_day_new] = rebalanced_portfolio
        initial_df_cash.loc[start_day:end_day_new] = cash_rebalanced

    df_port_sum = initial_port_df.sum(axis=1)
    df_port_sum = df_port_sum.mask(df_port_sum.eq(0)).ffill()
    df_sum_final = df_port_sum + initial_df_cash['Cash']

    df_final = df_sum_final.to_frame(name='Portfolio Value')

    return df_final, initial_port_df

def weights_rebalancing_main(df, weights, invest_amount, df_multiplied, initial_df_cash):

    initial_port_df = df.copy().astype(float)
    print('initial_port_df',initial_port_df)
    sorted_total_dates, stock_dates =  date_dict(initial_port_df) #return the sorted dates
    #print(sorted_total_dates)
    #print(sorted_total_dates)
    #print(stock_dates)
    for i, date in enumerate(sorted_total_dates):
        new_weights_bool = bool_list(date, initial_port_df, stock_dates) #getting the boolean list of weights
        new_weights = rebalance_weights(weights, new_weights_bool) #rebalancing the weights
        start_day = sorted_total_dates[i] #assigning the start day and end day for each window of csv
        if i < (len(sorted_total_dates) - 1):
            end_day = sorted_total_dates[i+1]
            end_day_index = df.index.get_loc(end_day) - 1
            end_day_new = df.index[end_day_index]
        else:
            end_day_new = None

        '''
        if end_day_new is not None:
            period_port_df = df.loc[start_day:end_day_new]
            period_leverage_df = df_multiplied.loc[start_day:end_day_new]
        else:
            period_port_df = df.loc[start_day:]
            period_leverage_df = df_multiplied.loc[start_day:]
        '''        

        # Extract the relevant data for the current period
        period_port_df = df.loc[start_day:end_day_new] # slicing to the new window
        period_leverage_df = df_multiplied.loc[start_day:end_day_new]

        invest_amounts = [invest_amount * i for i in new_weights] #getting the initial amounts at the start

        rebalanced_portfolio = multi_portfolio(period_port_df, period_leverage_df, invest_amounts)
        invest_amount = rebalanced_portfolio.iloc[-1].sum() #reassigning the invest amount to sum of the last row

        
        initial_port_df.loc[start_day:end_day_new] = rebalanced_portfolio
        # Update the initial portfolio data with the rebalanced portfolio for the current period
        '''
        if end_day_new is not None:
            initial_port_df.loc[start_day:end_day_new] = rebalanced_portfolio
        else:
            initial_port_df.loc[start_day:] = rebalanced_portfolio
        '''



    df_port_sum = initial_port_df.sum(axis=1)



    #df_port_sum = df_port_sum.replace(to_replace=0, method='ffill')
    #df_port_sum = df_port_sum.mask(df_port_sum.eq(0)).ffill()


    df_sum_final = df_port_sum + initial_df_cash['Cash']


    df_final = df_sum_final.to_frame(name='Portfolio Value')
    #df_final = df_final.reset_index().rename(columns={"index":"Date"}) #resetting index



    return df_final, initial_port_df

def get_rebalance_indexlist(df, freq): # function to get index at which rebalancing needs to happen

    df_index = df.reset_index().rename(columns={"index":"Date"}) #resetting index

    #start_index = df_index.loc[df_index['Date'] == rebalance_date].index[0]

    start_index = 0

    end_index = df_index.index[-1]

    #rebalance_date_pd = pd.to_datetime(rebalance_date, format='%d-%m-%Y') # converting date to pandas datetime

    #end_date = port_df.index[-1]

    index_list = list(range(start_index, (end_index + 1) , freq))


    return index_list

def date_dict(df1): 
    # Function to get the start and end dates of each column
    stock_dates = {}
    start_dates_list = []
    end_dates_list = []

    df = df1.copy()

    for column in df.columns:
        # Find the first and last non-null index for each column
        start_date = df[column].first_valid_index()
        end_date = df[column].last_valid_index()
        
        if start_date is None or end_date is None:
            continue

        # Get the next date after the end date, as the new weights will start the next day after the end date
        current_index_position = df.index.get_loc(end_date)
        next_index_position = current_index_position + 1

        if next_index_position < len(df.index):  # Ensure the next date does not exceed the index
            next_end_date = df.index[next_index_position]
        else:
            next_end_date = end_date

        # Store the results in the dictionary with formatted dates
        stock_dates[column] = {
            'start_date': start_date.strftime('%d-%m-%Y'), 
            'end_date': end_date.strftime('%d-%m-%Y')
        } 
        start_dates_list.append(start_date.strftime('%d-%m-%Y'))
        end_dates_list.append(next_end_date.strftime('%d-%m-%Y'))

    # Combine, deduplicate, and sort the dates
    total_dates = start_dates_list + end_dates_list
    total_dates = list(set(total_dates))
    last_date = df.index[-1].strftime('%d-%m-%Y')
    
    if last_date in total_dates:
        total_dates.remove(last_date)

    sorted_total_dates = sort_dates(total_dates)  # Sort the total start and end dates
    print(stock_dates)
    return sorted_total_dates, stock_dates

def sort_dates(dates):
    # Sort the dates ensuring they are in datetime format
    # Convert strings back to datetime, sort, and reformat to 'dd-mm-yyyy'
    dates = [datetime.strptime(date, '%d-%m-%Y') for date in dates]
    return [date.strftime('%d-%m-%Y') for date in sorted(dates)]

def bool_list(date, df, stock_dates):  
    # Function that returns the boolean list for weights: True for columns that have weights, False otherwise
    date = convert_to_datetime(date)  # Ensure date is in datetime format

    weight_bool_list = []
    
    for column in df.columns:
        if df[column].isnull().all():
            weight_bool_list.append(False)
            continue

        # Convert start and end dates back to datetime for comparison
        start_date_col = datetime.strptime(stock_dates[column]['start_date'], '%d-%m-%Y')
        end_date_col = datetime.strptime(stock_dates[column]['end_date'], '%d-%m-%Y')

        # Creating a list of booleans: True if the weight should be included, False otherwise
        if (start_date_col > date) or (end_date_col < date):
            weight_bool_list.append(False)
        else:
            weight_bool_list.append(True)
    
    return weight_bool_list

def rebalance_weights(initial_weights, flag_list): #rebalancing weights based on the boolean list of weights
    print('initial_weights', initial_weights)
    sum=0
    count = 0
    sum_weights = 0

    final_weights = [None] * len(initial_weights)

    for i in range(len(final_weights)):
        
        if flag_list[i] == False:
            sum += initial_weights[i]
            final_weights[i] = 0    #assigning 0 weight to False

        else:
            count += 1
            sum_weights += initial_weights[i]


    if sum==0:
        return initial_weights

    for i in range(len(final_weights)):

        if flag_list[i] == True:
            final_weights[i] = initial_weights[i] + (initial_weights[i]*sum/ sum_weights) #assigning rebalanced weight to True


    return final_weights

def rebalancing2(df, df_multiplied, weights, freq, invest_amount, start_index, cash_percent, total_invest_amount, initial_df_cash, cash_amount_pre):
        
   
    initial_port_df = df.copy().astype(float)
    df_cash = initial_df_cash.copy()
    df_index = initial_port_df.reset_index().rename(columns={"index": "Date"})  # resetting index
    end_index = df_index.index[-1]
    
    ##print('Initial Portfolio DataFrame:\n', initial_port_df)
    ##print('Initial Cash DataFrame:\n', df_cash)
    ##print('DataFrame with reset index:\n', df_index)
    ##print('End index:', end_index)

    index_list = list(range(start_index, (end_index + 1), freq))  # indices for rebalancing
    ##print('Rebalancing indices:', index_list)

    df_cash.iloc[0:start_index, 0] = cash_amount_pre
    ##print('Cash DataFrame after setting pre-start index cash:\n', df_cash)

    initial_invest_amounts = [invest_amount * i for i in weights]  # initial amounts per weight
    ##print('Initial investment amounts based on weights:', initial_invest_amounts)
    
    period_port_df_pre = initial_port_df.iloc[0: start_index]
    period_leverage_df_pre = df_multiplied.iloc[0: start_index]
    ##print('Period Portfolio DataFrame before rebalancing:\n', period_port_df_pre)
    ##print('Pre-Period Leverage DataFrame:\n', period_leverage_df_pre)
    
    rebalanced_portfolio_pre = multi_portfolio(period_port_df_pre, period_leverage_df_pre, initial_invest_amounts)
    ##print('Rebalanced Portfolio DataFrame before frequency rebalancing:\n', rebalanced_portfolio_pre)

    invest_amount = rebalanced_portfolio_pre.iloc[-1].sum()
    ##print('Investment amount after rebalancing:\n', invest_amount)

    total_invest_amount = invest_amount + cash_amount_pre
    ##print('cash_amount_pre:', cash_amount_pre)
    ##print('Total investment amount including cash:\n', total_invest_amount)

    initial_port_df.iloc[0: start_index] = rebalanced_portfolio_pre

    # Rebalancing at specified frequency
    for rebalance_index in index_list:
        ##print('\nRebalancing at index:', rebalance_index)

        if rebalance_index == end_index:
            cash_amount_reb = total_invest_amount * cash_percent
            invest_amount = total_invest_amount - cash_amount_reb
            df_cash.iloc[rebalance_index, 0] = cash_amount_reb
            ##print('Final rebalancing at end index. Cash rebalance amount:', cash_amount_reb)
            ##print('Investment amount after cash rebalance:', invest_amount)
            ##print('weightsasaas', weights)
            rebalance_initial = [invest_amount * i for i in weights]
            rebalance_final = rebalance_initial * (1 + df_multiplied.iloc[rebalance_index])
            initial_port_df.iloc[rebalance_index] = rebalance_final

            ##print('Final rebalanced portfolio values:\n', initial_port_df.iloc[rebalance_index])
            return initial_port_df, df_cash

        cash_amount_reb = total_invest_amount * cash_percent
        invest_amount = total_invest_amount - cash_amount_reb
        df_cash.iloc[rebalance_index: rebalance_index + freq, 0] = cash_amount_reb
        ##print('Cash rebalance amount at index:', rebalance_index, 'is:', cash_amount_reb)
        ##print('weightsasaas ', weights)
        rebalance_initial = [invest_amount * i for i in weights]
        ##print('Initial investment amounts for rebalancing:', rebalance_initial)

        rebalance_final = rebalance_initial * (1 + df_multiplied.iloc[rebalance_index])  # rebalance calculation
        initial_port_df.iloc[rebalance_index] = rebalance_final
        ##print('Rebalanced portfolio values at index:', rebalance_index, '\n', rebalance_final)

        invest_amounts = initial_port_df.iloc[rebalance_index].values.flatten().tolist()
        period_port_df = initial_port_df.iloc[rebalance_index + 1: rebalance_index + freq]
        period_leverage_df = df_multiplied.iloc[rebalance_index + 1: rebalance_index + freq]
        
        ##print('Period Portfolio DataFrame after rebalance index:\n', period_port_df)
        ##print('Period Leverage DataFrame after rebalance index:\n', period_leverage_df)

        rebalanced_portfolio = multi_portfolio(period_port_df, period_leverage_df, invest_amounts)
        ##print('Rebalanced Portfolio DataFrame for the period:\n', rebalanced_portfolio)

        if rebalance_index + 1 == rebalance_index + freq:  # daily frequency rebalance condition
            invest_amount = initial_port_df.iloc[rebalance_index].sum()
        else:
            invest_amount = rebalanced_portfolio.iloc[-1].sum()
        ##print('Investment amount after rebalancing:\n', invest_amount)

        total_invest_amount = invest_amount + cash_amount_reb
        ##print('Total investment amount after rebalancing including cash:\n', total_invest_amount)

        initial_port_df.iloc[rebalance_index + 1: rebalance_index + freq] = rebalanced_portfolio

    ##print(f'Final Portfolio DataFrame after rebalancing:\n', initial_port_df)
    ##print(f'Final Cash DataFrame after rebalancing:\n', df_cash)

    return initial_port_df, df_cash

def multi_portfolio(df, df_multiplied, invest_amounts):
    df_port = df.copy()

    cumulative_product = (1 + df_multiplied).cumprod()

    df_port = cumulative_product * invest_amounts

    return df_port

def extract_system_names(selected_systems):
    # Ensure all items are dictionaries with the expected key
    system_names = []
    for system in selected_systems:
        if isinstance(system, dict) and 'system' in system:
            system_names.append(system['system'])
        
    return system_names

def extract_debtfund_names(debtfund_systems):
    # Ensure all items are dictionaries with the expected key 'debtfund'
    debtfund_names = []
    for debtfund in debtfund_systems:
        if isinstance(debtfund, dict) and 'debtfund' in debtfund:
            debtfund_names.append(debtfund['debtfund'])  # Corrected to extract 'debtfund' names
        
    return debtfund_names

def calculate_weights(selected_systems):
    # Convert weightage to float before dividing by 100
    return [float(system['weightage']) / 100 for system in selected_systems]

def validate_selected_systems(selected_systems):
    if not isinstance(selected_systems, list):
        raise ValueError("selected_systems must be a list")
    for system in selected_systems:
        if not isinstance(system, dict) or 'system' not in system or 'weightage' not in system:
            raise ValueError("Each system must be a dictionary with 'system' and 'weightage' keys")

def initialize_cash_df(df, cash):

    df2 = pd.DataFrame(index=df.index, columns=['Cash'])
    df2['Cash'] = cash
    
    return df2

def percentage_df_fun(df, lev): # function to get the PL along with leverage
    ##print('df',df)
    ##print('leverages',lev)
    df_percentage_change = df.pct_change(fill_method=None).fillna(0) #calculating percentage change
    
    df_multiplied = df_percentage_change.copy()

    #multiplying the each system percentage change by the leverage

    for col, multiplier in zip(df_percentage_change.columns, lev):
        df_multiplied[col] *= multiplier
    

    
    ##print('df_multiplied',df_multiplied)

    return df_multiplied

def calculate_leverage_values(selected_systems):
    return [system['leverage'] for system in selected_systems]

def weights_rebalance_frequency_debt(df, weights, df_multiplied, freq, invest_amount):
    print('weights_rebalance_frequency_debt',df)
    initial_port_df = df.copy().astype(float)
    sorted_total_dates, stock_dates = date_dict(initial_port_df)  # return the sorted dates
    
    # Convert the DataFrame index to datetime if not already
    initial_port_df.index = pd.to_datetime(initial_port_df.index, format='%d-%m-%Y')
    df_multiplied.index = pd.to_datetime(df_multiplied.index, format='%d-%m-%Y')
    print('sorted_total_dates', sorted_total_dates)
    
    for i, date in enumerate(sorted_total_dates):
        # Convert date to datetime for matching
        date = pd.to_datetime(date, format='%d-%m-%Y')
        
        new_weights_bool = bool_list(date, initial_port_df, stock_dates)  # getting the boolean list of weights
        new_weights = rebalance_weights(weights, new_weights_bool)  # rebalancing the weights
        start_day = date
        print(f"Processing start day: {start_day}")  # Debug print
        
        if i < (len(sorted_total_dates) - 1):
            end_day = pd.to_datetime(sorted_total_dates[i+1], format='%d-%m-%Y')
            end_day_index = initial_port_df.index.get_loc(end_day) - 1
            end_day_new = initial_port_df.index[end_day_index]
        else:
            end_day_new = None
        
        date_index = initial_port_df.index.get_loc(start_day)
        start_index = freq - (date_index % freq)  # calculating the start index of the rebalancing for the new sliced dataframe
        
        # Extract the relevant data for the current period
        period_port_df = initial_port_df.loc[start_day:end_day_new]  # slicing to the new window
        period_leverage_df = df_multiplied.loc[start_day:end_day_new]
        
        # rebalanced_portfolio
        rebalanced_portfolio = rebalancing_debt(period_port_df, period_leverage_df, new_weights, freq, start_index, invest_amount)
        
        invest_amount = rebalanced_portfolio.iloc[-1].sum()
        # Update the initial portfolio data with the rebalanced portfolio for the current period
        initial_port_df.loc[start_day:end_day_new] = rebalanced_portfolio
    
    df_port_sum = initial_port_df.sum(axis=1)
    df_port_sum = df_port_sum.mask(df_port_sum.eq(0)).ffill()
    
    # making the dataseries a dataframe to make it easier to plot
    df_final = df_port_sum.to_frame(name='Portfolio Value')
    print('df_final',df_final)
    return df_final

def rebalancing_debt(df, df_multiplied, weights, freq, start_index, invest_amount):
    initial_port_df = df.copy().astype(float)

    df_index = initial_port_df.reset_index().rename(columns={"index":"Date"}) #resetting index


    end_index = df_index.index[-1]

    #rebalance_date_pd = pd.to_datetime(rebalance_date, format='%d-%m-%Y') # converting date to pandas datetime

    #end_date = port_df.index[-1]

    index_list = list(range(start_index, (end_index + 1) , freq)) # getting the list of index which need to be rebalanced at



    initial_invest_amounts = [invest_amount * i for i in weights] # getting the start amount for each column based on weights
    period_port_df_pre = initial_port_df.iloc[0: start_index]
    period_leverage_df_pre = df_multiplied.iloc[0: start_index]

    rebalanced_portfolio_pre = multi_portfolio(period_port_df_pre, period_leverage_df_pre, initial_invest_amounts)

    invest_amount = rebalanced_portfolio_pre.iloc[-1].sum()

        # Update the initial portfolio data with the rebalanced portfolio for the current period
    initial_port_df.iloc[0: start_index] = rebalanced_portfolio_pre

    
    for rebalance_index in index_list:
        
        if rebalance_index == end_index:

            rebalance_initial = [invest_amount * i for i in weights]
            rebalance_final = rebalance_initial * (1+ df_multiplied.iloc[rebalance_index])
            initial_port_df.iloc[rebalance_index] = rebalance_final


            return initial_port_df
        


        ###print(f'Total Invest amount: {total_invest_amount}')


        ###print(f'Invest amount: {invest_amount}')

        rebalance_initial = [invest_amount * i for i in weights]

        ###print(f'Rebalance initial: {rebalance_initial}')

        rebalance_final = rebalance_initial * (1+ df_multiplied.iloc[rebalance_index]) # calculating value at the frequency date
        initial_port_df.iloc[rebalance_index] = rebalance_final

        invest_amounts = initial_port_df.iloc[rebalance_index].values.flatten().tolist()


        period_port_df = initial_port_df.iloc[rebalance_index + 1: rebalance_index  + freq] #calculating value after frequecy date 
        period_leverage_df = df_multiplied.iloc[rebalance_index + 1 : rebalance_index + freq]


        rebalanced_portfolio = multi_portfolio(period_port_df, period_leverage_df, invest_amounts)



        if rebalance_index + 1 == rebalance_index  + freq: # if condition for daily frequency rebalance
            invest_amount = initial_port_df.iloc[rebalance_index].sum()

        else:
            invest_amount = rebalanced_portfolio.iloc[-1].sum()


        # Update the initial portfolio data with the rebalanced portfolio for the current period
        initial_port_df.iloc[rebalance_index + 1 : rebalance_index + freq] = rebalanced_portfolio

    




    ##print(f'Before: {initial_port_df}')

    return initial_port_df

def add_debtfund(df_sum_debt1, df_sum1): #function to add debtfund returns to the portfolio value
    df_sum_debt = df_sum_debt1.copy()
    if isinstance(df_sum1, tuple):
        df_sum1 = df_sum1[0]  # Unpack the first element if df_sum1 is a tuple
    # Proceed with the usual DataFrame operations
    df_sum = df_sum1.copy()

    # Ensure both indices are in the same datetime format and frequency
    df_sum.index = pd.to_datetime(df_sum.index, format='%d-%m-%Y')
    df_sum_debt.index = pd.to_datetime(df_sum_debt.index, format='%d-%m-%Y')

    # Reindex to ensure both dataframes have matching indices (dates)
    df_sum_debt = df_sum_debt.reindex(df_sum.index, method='ffill')

    initial_value = df_sum['Portfolio Value'].iloc[0]

    # Handle percentage change and cumulative product
    df_percentage_change_debt = df_sum_debt['Portfolio Value'].pct_change().fillna(0) #calculating the percentage change
    df_percentage_change_port = df_sum['Portfolio Value'].pct_change().fillna(0)
    df_percentage_change = df_percentage_change_debt.add(df_percentage_change_port, fill_value=0)

    cumulative_product = (1 + df_percentage_change).cumprod() #taking the cumulative product
    df_port = cumulative_product * initial_value #multiplying cumulative product with intial amount

    df_port = df_port.to_frame(name='Portfolio Value')

    df_port.index = df_port.index.strftime('%d-%m-%Y')  # Ensure date formatting is consistent
    print('add_debtfund',df_port)

    return df_port

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
