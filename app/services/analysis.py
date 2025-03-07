# app/services/analysis.py
import os
import traceback
from flask import Flask, request, jsonify, make_response, url_for, current_app
import pandas as pd
import numpy as np
import logging
from pyxirr import xirr
from app.services.data_processing import load_data
from app.services.database import get_db_connection
from app.utils.helpers import convert_nan_to_none
from datetime import datetime
import numpy_financial as npf
from app.config import Config
import os


def standardize_db_data(df_db):
    """
    Standardizes database DataFrame to have indices as columns.
    """
    try:
        # Log input data
        logging.info(f"Input DB DataFrame shape: {df_db.shape}")
        logging.info(f"Unique indices in DB: {df_db['indices'].unique()}")
        
        # Pivot the data to convert indices to columns
        df_standardized = df_db.pivot_table(
            index='date',
            columns='indices',
            values='nav',
            aggfunc='last'  # Use last value if there are duplicates
        ).reset_index()
        
        # Log the shape after pivot
        logging.info(f"DB data after pivot - shape: {df_standardized.shape}")
        logging.info(f"DB data date range: {df_standardized['date'].min()} to {df_standardized['date'].max()}")
        
        return df_standardized
    
    except Exception as e:
        logging.error(f"Error in standardize_db_data: {str(e)}")
        raise

def standardize_csv_data(df_csv):
    """
    Standardizes CSV DataFrame format if needed.
    """
    logging.info(f"Input CSV DataFrame shape: {df_csv.shape}")
    logging.info(f"CSV data date range: {df_csv['date'].min()} to {df_csv['date'].max()}")
    
    # Rename date column to match DB format
    df_csv = df_csv.rename(columns={'date': 'date'})
    
    return df_csv

# app/services/analysis.py

def standardize_strategy_names(df):
    """
    Standardizes strategy names by removing spaces, handling _1 suffixes properly.
    
    Args:
        df (pd.DataFrame): Input DataFrame with strategy columns
    
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    renamed_columns = {}
    for col in df.columns:
        if col.lower() == 'date':
            continue
            
        # Remove extra spaces and standardize
        new_name = col.strip()
        # Preserve _1 suffix if it exists
        if '_1' in new_name:
            base_name = new_name.replace('_1', '')
            new_name = f"{base_name.strip()}_1"
        
        renamed_columns[col] = new_name
    
    return df.rename(columns=renamed_columns)

def load_default_strategies():
    """
    Fetches and combines strategies from the database, default CSV, and user-uploaded CSV files,
    properly handling historical data and strategy name standardization.
    """
    try:
        # 1. Load database data
        conn = get_db_connection()
        query = """
            SELECT DISTINCT ON (date, indices) 
                date, 
                indices, 
                nav
            FROM tblresearch 
            ORDER BY date, indices, updated_at DESC;
        """
        df_db = pd.read_sql_query(query, conn)
        conn.close()
        
        logging.info(f"Fetched {len(df_db)} rows from PostgreSQL")
        
        # 2. Load default CSV data
        default_csv_path = os.path.join(Config.CSV_DIR, 'All weather portfolio.csv')
        df_csv = pd.read_csv(default_csv_path)
        logging.info(f"Loaded {len(df_csv)} rows from default CSV")

        # 3. Standardize both datasets
        df_db_standardized = standardize_db_data(df_db)
        df_csv_standardized = standardize_csv_data(df_csv)
        
        # Ensure dates are in datetime format
        df_db_standardized['date'] = pd.to_datetime(df_db_standardized['date'])
        df_csv_standardized['date'] = pd.to_datetime(df_csv_standardized['date'])
        
        # 4. Create initial combined DataFrame
        combined_df = pd.concat([df_db_standardized, df_csv_standardized])
        combined_df = combined_df.sort_values('date')
        combined_df = combined_df.groupby('date').last().reset_index()
        
        # 5. Process uploaded CSV files
        upload_dir = os.path.join(current_app.root_path, 'uploads')
        if os.path.exists(upload_dir):
            uploaded_files = [f for f in os.listdir(upload_dir) if f.lower().endswith('.csv')]
            
            for filename in uploaded_files:
                file_path = os.path.join(upload_dir, filename)
                try:
                    df_uploaded = pd.read_csv(file_path)
                    logging.info(f"Processing uploaded CSV: {filename}")
                    logging.info(f"Original columns in {filename}: {df_uploaded.columns.tolist()}")
                    
                    # Standardize column names
                    df_uploaded = standardize_strategy_names(df_uploaded)
                    logging.info(f"Standardized columns in {filename}: {df_uploaded.columns.tolist()}")
                    
                    # Convert date column
                    date_col = [col for col in df_uploaded.columns if col.lower() == 'date'][0]
                    df_uploaded[date_col] = pd.to_datetime(df_uploaded[date_col], errors='coerce')
                    df_uploaded = df_uploaded.rename(columns={date_col: 'date'})
                    
                    # Drop rows with NaT dates
                    df_uploaded = df_uploaded.dropna(subset=['date'])
                    
                    # Get non-date columns
                    strategy_columns = [col for col in df_uploaded.columns if col.lower() != 'date']
                    
                    # Merge with combined_df
                    combined_df = pd.merge(
                        combined_df,
                        df_uploaded[['date'] + strategy_columns],
                        on='date',
                        how='outer'
                    )
                    
                    logging.info(f"Successfully merged {filename}")
                    logging.info(f"Current columns after merge: {combined_df.columns.tolist()}")
                    
                except Exception as e:
                    logging.error(f"Error processing uploaded CSV {filename}: {str(e)}", exc_info=True)
        
        # 6. Final sorting and cleaning
        combined_df = combined_df.sort_values('date')
        
        # Log final results
        logging.info(f"Final combined shape: {combined_df.shape}")
        logging.info(f"Final date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logging.info(f"Final columns: {combined_df.columns.tolist()}")
        
        return combined_df

    except Exception as e:
        logging.error(f"Error loading strategies: {str(e)}", exc_info=True)
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def calculate_annual_returns(df):
    """
    Calculate annual returns from the portfolio equity curve and return as key-value pairs.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'date' and 'Portfolio Value' columns
    
    Returns:
    list: List of dictionaries with 'year' and 'return' keys
    """
    # Convert date column to datetime with European format (DD-MM-YYYY)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    # Set date as index
    df_indexed = df.set_index('date')
    
    # Get daily returns
    daily_returns = df_indexed['Portfolio Value'].pct_change()
    
    # Group by year and calculate compound annual returns
    annual_returns = (1 + daily_returns).groupby(daily_returns.index.year).prod() - 1
    
    # Convert to percentage and round to 2 decimal places
    annual_returns = (annual_returns * 100).round(2)
    
    # Convert to list of dictionaries with year and return keys
    result = [{'year': str(year), 'return': float(return_val)} 
             for year, return_val in annual_returns.items()]
    
    # Sort by year
    result.sort(key=lambda x: x['year'])
    
    return result

def calculate_trailing_returns(df_sum):
    """
    Calculate trailing returns for specified periods:
    - For periods >= 1 year: Uses CAGR (Compound Annual Growth Rate)
    - For periods < 1 year: Uses absolute returns
    
    Args:
        df_sum (pd.DataFrame): DataFrame with portfolio values and date index
        
    Returns:
        dict: Dictionary containing trailing returns for each period
    """
    import logging
    import pandas as pd
    import numpy as np
    
    logging.info(f"Input DataFrame shape: {df_sum.shape}")
    logging.info(f"Input DataFrame columns: {df_sum.columns}")
    
    # Ensure we're working with a DataFrame with datetime index
    if 'date' in df_sum.columns:
        logging.info("Converting 'date' column to index")
        df = df_sum.set_index('date')
    else:
        df = df_sum.copy()
        
    logging.info(f"DataFrame index type: {type(df.index)}")
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            logging.info("Converting index to datetime")
            df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')
            logging.info(f"Conversion successful. New index type: {type(df.index)}")
        except Exception as e:
            logging.error(f"Error converting dates to datetime: {e}")
            raise
    
    # Drop rows with invalid dates and log the count
    invalid_dates = df.index.isna().sum()
    if invalid_dates > 0:
        logging.warning(f"Found {invalid_dates} invalid dates in the index. Dropping these rows.")
        df = df.dropna(axis=0, subset=[df.index.name] if df.index.name else None)
    
    # Verify portfolio_value column
    if 'Portfolio Value' not in df.columns:
        logging.error("'Portfolio Value' column not found. Available columns: " + ", ".join(df.columns))
        return {}
    
    # Get date range info
    latest_date = df.index.max()
    earliest_date = df.index.min()
    logging.info(f"Date range: {earliest_date} to {latest_date}")
    
    # Define periods and their corresponding days
    period_days = {
        '10d': 10,
        '1w': 7,
        '1m': 30,
        '3m': 91,
        '6m': 182,
        '1y': 365,
        '3y': 1095,
        '5y': 1825
    }
    
    trailing_returns = {}
    
    # Calculate returns for each period
    for period, days in period_days.items():
        try:
            start_date = latest_date - pd.Timedelta(days=days)
            
            if start_date < earliest_date:
                logging.info(f"Skipping {period} calculation - insufficient history")
                trailing_returns[period] = None
                continue
            
            # Find the closest available date
            start_value_series = df['Portfolio Value'].loc[
                (df.index >= start_date) & 
                (df.index <= start_date + pd.Timedelta(days=1))
            ]
            
            if start_value_series.empty:
                logging.warning(f"No data found for {period} start date {start_date}")
                trailing_returns[period] = None
                continue
            
            start_value = start_value_series.iloc[0]
            end_value = df['Portfolio Value'].iloc[-1]
            
            logging.debug(f"{period} - Start value: {start_value}, End value: {end_value}")
            
            # Calculate return based on period length
            if period in ['1y', '3y', '5y']:
                # Calculate CAGR for periods >= 1 year
                years = days / 365.0
                cagr = (((end_value / start_value) ** (1 / years)) - 1) * 100
                trailing_returns[period] = round(cagr, 2)
                logging.info(f"Successfully calculated {period} CAGR: {trailing_returns[period]}%")
            else:
                # Calculate absolute return for periods < 1 year
                abs_return = ((end_value / start_value) - 1) * 100
                trailing_returns[period] = round(abs_return, 2)
                logging.info(f"Successfully calculated {period} absolute return: {trailing_returns[period]}%")
                
        except Exception as e:
            logging.error(f"Error calculating {period} trailing return: {str(e)}")
            trailing_returns[period] = None
    
    logging.info(f"Final trailing returns: {trailing_returns}")
    return trailing_returns

def process_single_portfolio(data, df):
   
    try:
        # Validate the required fields
        required_fields = ['invest_amount', 'cash_percent', 'start_date', 'end_date', 'selected_systems', 'frequency']
        for field in required_fields:
            if field not in data:
                error_msg = f"Missing required field: {field}"
                logging.error(error_msg)
                return {"error": error_msg}

        # Extract data and parse dates
        invest_amount = data['invest_amount']
        cash_percent = data['cash_percent'] / 100
        start_date = pd.to_datetime(data['start_date'], format='%d-%m-%Y', errors='coerce')
        end_date = pd.to_datetime(data['end_date'], format='%d-%m-%Y', errors='coerce')
   
        if pd.isnull(start_date) or pd.isnull(end_date):
            error_msg = "Invalid start or end date provided."
            logging.error(error_msg)
            return {"error": error_msg}
        if start_date > end_date:
            error_msg = "Start date cannot be after end date."
            logging.error(error_msg)
            return {"error": error_msg}

        selected_systems_list = data['selected_systems']
        selected_systems_names = extract_system_names(selected_systems_list)
        selected_debtfund_list = data.get('selected_debtfunds', [])
        selected_debtfund_names = extract_debtfund_names(selected_debtfund_list)
        frequency = data['frequency']
        weights_systems = calculate_weights(selected_systems_list)
        leverage_systems = calculate_leverage_values(selected_systems_list)
        freq = get_frequency_value(frequency)
        
        new_DF = required_df(df, start_date, end_date, selected_systems_names)
        print('new_Ddcfwerfr4F', new_DF)
        # Check if any valid data exists for the selected date range and systems
        if new_DF.empty:
            error_msg = "No data available for the selected systems in the specified date range."
            logging.error(error_msg)
            return {"error": error_msg}
            
        # Check for at least one non-NaN value in each selected system
        valid_systems = new_DF.apply(lambda x: x.notna().any())
        invalid_systems = [sys for sys, valid in valid_systems.items() if not valid]
        
        if invalid_systems:
            error_msg = f"No valid data for the following systems: {', '.join(invalid_systems)}"
            logging.error(error_msg)
            return {"error": error_msg}

        logging.debug(f'New DF: {new_DF}')
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
            
            # Check if debt fund data is available
            if df_debts.empty:
                error_msg = "No data available for the selected debt funds in the specified date range."
                logging.error(error_msg)
                return {"error": error_msg}
                
            # Check for valid debt fund data
            valid_debt_funds = df_debts.apply(lambda x: x.notna().any())
            invalid_debt_funds = [fund for fund, valid in valid_debt_funds.items() if not valid]
            
            if invalid_debt_funds:
                error_msg = f"No valid data for the following debt funds: {', '.join(invalid_debt_funds)}"
                logging.error(error_msg)
                return {"error": error_msg}

            logging.debug(f"Debt funds dataframe:\n{df_debts.head()}")
            logging.debug(f"Debt funds dataframe shape: {df_debts.shape}")

            debt_leverage_values = calculate_leverage_values(selected_debtfund_list)
            weights_debts = calculate_weights(selected_debtfund_list)

            logging.debug(f"Debt leverage values: {debt_leverage_values}")
            logging.debug(f"Debt weights: {weights_debts}")

            df_multiplied_debt = percentage_df_fun(df_debts, debt_leverage_values)
            logging.debug(f"Multiplied debt dataframe:\n{df_multiplied_debt.head()}")
            logging.debug(f"Multiplied debt dataframe shape: {df_multiplied_debt.shape}")

            df_sum_debt = weights_rebalance_frequency_debt(
                df_debts, weights_debts, df_multiplied_debt, freq, new_inv_amount
            )
            logging.debug(f"Final debt sum dataframe:\n{df_sum_debt.head()}")
            logging.debug(f"Final debt sum dataframe shape: {df_sum_debt.shape}")

            df_sum = add_debtfund(df_sum_debt, df_sum)
            logging.debug(f"Combined dataframe after adding debt funds:\n{df_sum.head()}")
            logging.debug(f"Combined dataframe shape: {df_sum.shape}")
            logging.info("=== DEBT FUND PROCESSING END ===")

        validate_selected_systems(selected_systems_list)

        if 'date' not in df_sum.columns:
            df_sum = df_sum.reset_index().rename(columns={'index': 'date'})

        logging.debug(f"Final analysis dataframe:\n{df_sum.head()}")

        try:
            print('reigjtrejijhgpi4jp5jp', df_sum)
            analysis_result = show_analysis(df_sum)
            analysis_result['trailing_returns'] = calculate_trailing_returns(df_sum)
            annual_returns = calculate_annual_returns(df_sum)
            analysis_result['annual_returns'] = annual_returns

            global global_equity_curve_data
            global_equity_curve_data = analysis_result.get('equity_curve_data', {})
            logging.info('show_analysis completed successfully')
            return {
                'result': analysis_result,
            }
        except Exception as analysis_error:
            logging.error(f"Error in show_analysis: {str(analysis_error)}")
            logging.error(traceback.format_exc())
            return {
                "error": f"Error in analysis: {str(analysis_error)}",
                "traceback": traceback.format_exc()
            }

    except Exception as e:
        logging.error(f"Error in process_single_portfolio: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
                
def calculate_portfolio_comparison(data, combined_strategies_df):
    try:
        # Get the list of portfolios from the frontend request
        portfolios = data.get('portfolios', [])
        logging.info(f"Portfolios received for comparison: {portfolios}")

        # Initialize an empty list to store results
        results = []

        # Check if portfolios are provided
        if not portfolios:
            error_msg = "No portfolios found in the request."
            logging.error(error_msg)
            return [{"error": error_msg}]

        # Validate and prepare the combined_strategies_df
        if combined_strategies_df is None or combined_strategies_df.empty:
            error_msg = "No valid data provided in combined_strategies_df."
            logging.error(error_msg)
            return [{"error": error_msg}]

        # Prepare the DataFrame with proper date index
        df = combined_strategies_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]  # Remove rows with invalid dates

        # Ensure continuous date range for processing
        min_date = df.index.min()
        max_date = df.index.max()
        if pd.isnull(min_date) or pd.isnull(max_date):
            error_msg = "The date range contains invalid dates."
            logging.error(error_msg)
            return [{"error": error_msg}]

        df = df.loc[min_date:max_date].ffill()  # Fill forward for missing dates

        # Process each portfolio
        for idx, portfolio in enumerate(portfolios):
            logging.info(f"Processing portfolio {idx + 1}/{len(portfolios)}: {portfolio.get('name')}")
            
            # Retrieve the benchmark name for the portfolio
            benchmark_name = portfolio.get('benchmark')
            # if not benchmark_name:
            #     error_msg = f"Benchmark not specified for portfolio: {portfolio.get('name', 'Unnamed Portfolio')}"
            #     logging.error(error_msg)
            #     results.append({"error": error_msg, "portfolio_index": idx + 1})
            #     continue

            logging.info(f"Benchmark for portfolio '{portfolio.get('name')}': {benchmark_name}")

            # Get benchmark data using the `required_df` function
            # benchmark_data = required_df(df, min_date, max_date, [benchmark_name])
            # if benchmark_data.empty:
            #     error_msg = f"No data found for benchmark '{benchmark_name}' in the specified date range."
            #     logging.error(error_msg)
            #     results.append({"error": error_msg, "portfolio_index": idx + 1})
            #     continue

            # Process the portfolio
            portfolio_result = process_single_portfolio(portfolio, df)
            if 'error' in portfolio_result:
                results.append({
                    'portfolio_index': idx + 1,
                    'portfolio_name': portfolio.get('name', f'Portfolio {idx + 1}'),
                    'error': portfolio_result['error'],
                    'traceback': portfolio_result.get('traceback', ''),
                })
            else:
                # Add the benchmark comparison result
                # portfolio_result['result']['benchmark_data'] = benchmark_data.to_dict(orient='records')
                results.append({
                    'portfolio_index': idx + 1,
                    'portfolio_name': portfolio.get('name', f'Portfolio {idx + 1}'),
                    'selected_systems': portfolio['selected_systems'],
                
                    'benchmark_name': benchmark_name,
                    'result': portfolio_result['result'],
                })

        return results

    except Exception as e:
        logging.error(f"Error in calculate_portfolio_comparison: {str(e)}")
        logging.error(traceback.format_exc())
        return [{"error": str(e), "traceback": traceback.format_exc()}]

def calculate_annual_returns_with_balance(df_sum):
    print('calculate_annual_returns_with_balance', df_sum)
    """
    Calculate annual returns with balance for each year.
    
    Args:
        df_sum (pd.DataFrame): DataFrame with portfolio values and date index
        
    Returns:
        dict: Dictionary containing yearly returns and balances
    """
    # Ensure we're working with a DataFrame with datetime index
    if 'date' in df_sum.columns:
        df = df_sum.set_index('date')
    else:
        df = df_sum.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create empty dictionary to store results
    annual_metrics = {}
    
    try:
        # Get unique years in the data
        years = df.index.year.unique()
        
        for year in years:
            year_data = df[df.index.year == year]
            
            # Get first and last portfolio value for the year
            start_value = year_data['Portfolio Value'].iloc[0]
            end_value = year_data['Portfolio Value'].iloc[-1]
            
            # Calculate return percentage
            year_return = ((end_value / start_value) - 1) * 100
            
            # Store both return percentage and end balance
            annual_metrics[str(year)] = {
                'return_percentage': round(year_return, 2),
                'end_balance': end_value,
                'start_balance': round(start_value, 2),
                'start_date': year_data.index[0].strftime('%Y-%m-%d'),
                'end_date': year_data.index[-1].strftime('%Y-%m-%d')
            }
            
            # Calculate additional metrics
            highest_balance = year_data['Portfolio Value'].max()
            lowest_balance = year_data['Portfolio Value'].min()
            
            annual_metrics[str(year)].update({
                'highest_balance': round(highest_balance, 2),
                'lowest_balance': round(lowest_balance, 2),
                'drawdown': round(((highest_balance - lowest_balance) / highest_balance) * 100, 2)
            })
    
    except Exception as e:
        logging.error(f"Error calculating annual returns: {str(e)}")
        logging.error(traceback.format_exc())
        return None

    return annual_metrics

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
    print(df.info())
    cols = list(df.columns)
    cols = cols[1:]
    df1 = df[systems]
    print('start_date', start_date)
    print('end_date', end_date)
    print('systems', systems)
    print('lenght of df1', len(df1))
    if (start_date != '0' and end_date != '0'):
        print(df1.info())
        print('df1',df1)
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

    # Check if 'date' column exists and handle it appropriately
    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    
    # Convert 'date' to datetime without setting it as index
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Check for duplicates in 'date' and remove or aggregate them if necessary
    if df['date'].duplicated().any():
        df = df.drop_duplicates(subset='date', keep='last')
    
    # Sort the dataframe by date to ensure correct order
    df = df.sort_values('date')

    initial_index = df['Portfolio Value'].first_valid_index()
    initial_value = -df['Portfolio Value'].iloc[initial_index]
    final_value = df['Portfolio Value'].iloc[-1]

    print(f'Dataframe: {df}')
    print(f'Initial value: {initial_value}')
    print(f'Final Value: {final_value}')

    # Use the first and last date for XIRR calculation
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]

    car2 = xirr([start_date, end_date], [initial_value, final_value])  # using xirr function
    car2 = car2 * 100

    return car2

def validate_data(df):
    assert df['date'].dtype == 'datetime64[ns]', "date column should be datetime type"
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

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Drop any rows where the 'date' conversion failed (optional)
    df = df.dropna(subset=['date'])

    # Set 'date' as the index
    df.set_index('date', inplace=True)

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
                    'Peak_date': format_date(peak_date),
                    'Drawdown_date': format_date(trough_date),
                    'Recovery_date': format_date(recovery_date),
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
            'Peak_date': format_date(peak_date),
            'Drawdown_date': format_date(trough_date),
            'Recovery_date': 'Ongoing',
            'Drawdown': round(trough_value, 2),
            'Days': None  # Set to None for ongoing drawdowns
        })

    # Convert drawdown periods to DataFrame
    df_drawdown = pd.DataFrame(drawdown_periods)

    # Correctly calculate the "Days between Drawdown and Recovery date"
    df_drawdown['Days between Drawdown and Recovery date'] = df_drawdown.apply(
        lambda row: (pd.to_datetime(row['Recovery_date'], format='%d-%m-%Y') - pd.to_datetime(row['Drawdown_date'], format='%d-%m-%Y')).days 
        if row['Recovery_date'] != 'Ongoing' else pd.NA, axis=1)  # Set to pd.NA for ongoing drawdowns

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

    # Return all calculated metrics, including the updated df_drawdown with "Days between Drawdown and Recovery date"
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

    # Ensure 'date' is set as the index
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    
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
    # First, ensure the 'date' column is of string type (in case it's not)
    df['date'] = df['date'].astype(str)

    # Convert the 'date' column to datetime, coercing any invalid entries to NaT
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Check for any NaT values (which means invalid dates were found)
    invalid_dates = df[df['date'].isna()]
    if not invalid_dates.empty:
        print("Invalid dates found:", invalid_dates)

    # Now, you can safely set the 'date' column as the index if needed
    df.set_index('date', inplace=True)  


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

        # Ensure 'date' column is present and properly formatted
        if 'date' not in df_sum_new.columns:
            df_sum_new = df_sum_new.reset_index()
        if 'date' not in df_sum_new.columns:
            raise ValueError("The dataframe does not contain a 'date' column after reset_index.")

        # Convert 'date' column to datetime, assuming it's in 'dd-mm-yyyy' format
        df_sum_new['date'] = pd.to_datetime(df_sum_new['date'], format='%d-%m-%Y')

        # Format 'date' column back to 'dd-mm-yyyy'
        df_sum_new['date'] = df_sum_new['date'].dt.strftime('%d-%m-%Y')
        logging.info('date column processed successfully')

        # ============ Existing Statsbox Calls ============
        logging.info('About to call statsbox function')
        car, max_dd, avg_dd, carbymdd, max_gain, max_loss, top_10_worst_drawdowns, drawdown_table, max_peaktopeak = statsbox(df_sum_new)
        logging.info('statsbox function completed successfully')

        logging.info('About to call monthly_pl_table function')
        monthly_pl_pivot = monthly_pl_table(df_sum_new)
        logging.info('monthly_pl_table function completed successfully')

        logging.info('Analysis calculations completed')

        # Format equity curve data
        equity_curve_data = df_sum_new.apply(lambda row: {
            'date': row['date'],
            'NAV': row['Portfolio Value']
        }, axis=1).tolist()

        # Format drawdown data
        drawdown_data = drawdown_table.reset_index().apply(lambda row: {
            'date': row['date'].strftime('%d-%m-%Y') if pd.notnull(row['date']) else None,
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

        # ===================== NEW METRICS =====================
        # We'll do this after we've done the statsbox so that df_sum_new
        # is fully prepared. If you need to use the same df that has date as a
        # datetime, you might do so before re-formatting 'date' to string. 
        # For example, let's convert it back or recalculate:
        df_sum_new_dt = df_sum_new.copy()
        df_sum_new_dt['date'] = pd.to_datetime(df_sum_new_dt['date'], format='%d-%m-%Y', errors='coerce')

        additional_metrics = calculate_additional_metrics(df_sum_new_dt)
        additional_metrics['annual_metrics'] = calculate_annual_returns_with_balance(df_sum_new_dt)

        # ===================== Build Response =====================
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
            
            # Add the new metrics into the final JSON
            'additional_risk_return_metrics': additional_metrics,
        }

        # Convert NaNs to None for JSON serialization
        response = convert_nan_to_none(response)

        logging.info('Response prepared successfully')

        return response

    except Exception as e:
        logging.error(f"Error in show_analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def calculate_additional_metrics(df_sum, risk_free_rate=0.0, benchmark_returns=None):
    """
    Calculate additional risk/return metrics on a DataFrame that has
    a 'date' column and a 'Portfolio Value' column.

    :param df_sum: DataFrame with at least ['date', 'Portfolio Value'].
    :param risk_free_rate: Daily risk-free rate, annualized rate, or 0 if ignoring.
    :param benchmark_returns: (optional) A Series or DataFrame column of benchmark returns
                              aligned with df_sum['date'] for Beta, Alpha, etc.
    :return: A dict containing the calculated metrics.
    """

    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_sum['date']):
        df_sum['date'] = pd.to_datetime(df_sum['date'], format='%d-%m-%Y', errors='coerce')

    df_sum = df_sum.sort_values('date').reset_index(drop=True)

    # Daily Returns
    df_sum['Daily Return'] = df_sum['Portfolio Value'].pct_change()
    df_sum = df_sum.dropna(subset=['Daily Return'])

    # Annualized Return (CAGR)
    total_period_return = df_sum['Portfolio Value'].iloc[-1] / df_sum['Portfolio Value'].iloc[0] - 1
    days = (df_sum['date'].iloc[-1] - df_sum['date'].iloc[0]).days
    cagr = (1 + total_period_return) ** (365.0 / days) - 1 if days > 0 else np.nan

    # Standard Deviation (annualized)
    daily_std = df_sum['Daily Return'].std()
    ann_std = daily_std * np.sqrt(252)

    # Best Year / Worst Year Analysis
    df_sum['Year'] = df_sum['date'].dt.year
    yearly_returns = df_sum.groupby('Year')['Daily Return'].apply(lambda x: (1 + x).prod() - 1)

    if len(yearly_returns) > 0:
        best_year = yearly_returns.idxmax()
        worst_year = yearly_returns.idxmin()
        best_year_return = yearly_returns.max()
        worst_year_return = yearly_returns.min()
    else:
        best_year = worst_year = None
        best_year_return = worst_year_return = np.nan

    # Drawdown Analysis
    rolling_max = df_sum['Portfolio Value'].cummax()
    drawdown = (df_sum['Portfolio Value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # Minimum drawdown
    df_sum['Drawdown'] = drawdown

    # Average drawdown and average days in drawdown
    dd_groups = (drawdown != 0).astype(int).diff().ne(0).cumsum()
    dd_stats = df_sum.groupby(dd_groups).apply(
        lambda group: {'avg_dd': group['Drawdown'].mean(), 'days_in_dd': len(group)}
    ).values

    avg_dd = np.mean([stat['avg_dd'] for stat in dd_stats if not np.isnan(stat['avg_dd'])])
    avg_days_in_dd = np.mean([stat['days_in_dd'] for stat in dd_stats if stat['days_in_dd'] > 0])

    # Benchmark Correlation
    if benchmark_returns is not None:
        merged = pd.merge(
            df_sum[['date', 'Daily Return']],
            benchmark_returns.rename('Benchmark Return'),
            on='date',
            how='inner'
        )
        benchmark_corr = merged[['Daily Return', 'Benchmark Return']].corr().iloc[0, 1]

        # Beta
        cov = np.cov(merged['Daily Return'], merged['Benchmark Return'])[0, 1]
        var_bench = np.var(merged['Benchmark Return'])
        beta = cov / var_bench if var_bench != 0 else np.nan

        # Alpha (annualized)
        daily_alpha_series = merged['Daily Return'] - (risk_free_rate + beta * (merged['Benchmark Return'] - risk_free_rate))
        daily_alpha = daily_alpha_series.mean()
        alpha = daily_alpha * 252  # Annualized Alpha
    else:
        benchmark_corr = np.nan
        beta = 1.0
        alpha = 0.0

    # Sharpe Ratio
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1 if risk_free_rate > 0 else 0.0
    df_sum['Excess Daily Return'] = df_sum['Daily Return'] - daily_rf
    sharpe_ratio = (df_sum['Excess Daily Return'].mean() / df_sum['Excess Daily Return'].std()) * np.sqrt(252) \
        if df_sum['Excess Daily Return'].std() > 0 else np.nan

    # Sortino Ratio
    df_sum['Downside Return'] = np.where(df_sum['Daily Return'] < daily_rf, df_sum['Daily Return'] - daily_rf, 0)
    sortino_ratio = (df_sum['Excess Daily Return'].mean() / df_sum['Downside Return'].std()) * np.sqrt(252) \
        if df_sum['Downside Return'].std() > 0 else np.nan

    # Treynor Ratio
    treynor_ratio = (cagr - risk_free_rate) / beta if beta != 0 else np.nan

    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        'Annualized Return (CAGR)': cagr,
        'Standard Deviation (annualized)': ann_std,
        'Best Year': best_year,
        'Best Year Return': best_year_return,
        'Worst Year': worst_year,
        'Worst Year Return': worst_year_return,
        'Maximum Drawdown': max_drawdown,
        'Average Drawdown': avg_dd,
        'Average Days in Drawdown': avg_days_in_dd,
        'Benchmark Correlation': benchmark_corr,
        'Beta': beta,
        'Alpha (annualized)': alpha,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Treynor Ratio (%)': treynor_ratio * 100 if not np.isnan(treynor_ratio) else np.nan,
        'Calmar Ratio': calmar_ratio,
    }

    return metrics

def calculate_xirr(df1):
    df = df1.copy()

    # First, ensure the 'date' column is of string type (in case it's not)
    df['date'] = df['date'].astype(str)

    # Convert the 'date' column to datetime, coercing any invalid entries to NaT
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Check for any NaT values (which means invalid dates were found)
    invalid_dates = df[df['date'].isna()]
    if not invalid_dates.empty:
        print("Invalid dates found:", invalid_dates)

    # Now, you can safely set the 'date' column as the index if needed
    df.set_index('date', inplace=True)

    df = df.sort_index()

    df_resampled = df.resample('D').ffill()
    df_resampled = df_resampled.reset_index().rename(columns={'index': 'date'})

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

    # Convert the 'date' column in the df_resampled to datetimeIndex and format it
    new_df.index = pd.to_datetime(df_resampled['date']).dt.strftime('%d-%m-%Y')

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

def date_dict(df1): 
    """
    Get the start and end dates for each column (strategy) in the DataFrame.
    Returns sorted dates list and a dictionary with date ranges for each strategy.
    """
    try:
        # Validate input
        if df1 is None or df1.empty:
            logging.error("Input DataFrame is empty or None")
            raise ValueError("Input DataFrame cannot be empty")

        df = df1.copy()
        
        # Ensure we have data to process
        if len(df.index) == 0:
            logging.error("DataFrame has no rows")
            raise ValueError("DataFrame has no rows")
            
        if len(df.columns) == 0:
            logging.error("DataFrame has no columns")
            raise ValueError("DataFrame has no columns")
        
        # Initialize containers
        stock_dates = {}
        total_dates = set()

        # Process each column (strategy)
        for column in df.columns:
            # Get non-null values for this strategy
            strategy_data = df[column].dropna()
            
            if not strategy_data.empty:
                # Find first and last non-null index for this strategy
                start_date = strategy_data.index[0]
                end_date = strategy_data.index[-1]
                
                # Format dates
                start_date_str = start_date.strftime('%d-%m-%Y')
                end_date_str = end_date.strftime('%d-%m-%Y')
                
                # Store dates
                stock_dates[column] = {
                    'start_date': start_date_str,
                    'end_date': end_date_str
                }
                
                # Add to total dates set
                total_dates.add(start_date_str)
                total_dates.add(end_date_str)
                
                # Try to add next date after end date
                try:
                    end_loc = df.index.get_loc(end_date)
                    if end_loc + 1 < len(df.index):
                        next_date = df.index[end_loc + 1]
                        total_dates.add(next_date.strftime('%d-%m-%Y'))
                except Exception as e:
                    logging.warning(f"Could not add next date for {column}: {str(e)}")

        # Ensure we found some dates
        if not total_dates:
            raise ValueError("No valid dates found in DataFrame")

        # Convert to sorted list
        sorted_total_dates = sort_dates(list(total_dates))
        
        logging.info(f"Processed {len(stock_dates)} strategies with valid data")
        logging.info(f"date range: {sorted_total_dates[0]} to {sorted_total_dates[-1]}")
        
        return sorted_total_dates, stock_dates

    except Exception as e:
        logging.error(f"Error in date_dict: {str(e)}")
        raise

def weights_rebalance_frequency(df, weights, invest_amount, df_multiplied, freq, cash_percent, initial_df_cash, total_invest_amount):
    try:
        # Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")
            
        # Create copies of input data
        initial_port_df = df.copy().astype(float)
        cash_amount_end = total_invest_amount * cash_percent

        # Convert indices to datetime
        for frame in [initial_port_df, initial_df_cash, df_multiplied]:
            frame.index = pd.to_datetime(frame.index, format='%d-%m-%Y')

        # Get date information
        sorted_total_dates, stock_dates = date_dict(initial_port_df)
        
        if not sorted_total_dates:
            raise ValueError("No valid dates found for rebalancing")

        # Process each date
        for i, date in enumerate(sorted_total_dates):
            date = pd.to_datetime(date, format='%d-%m-%Y')
            
            # Calculate new weights
            new_weights_bool = bool_list(date, initial_port_df, stock_dates)
            new_weights = rebalance_weights(weights, new_weights_bool)
            
            # Determine period start and end
            start_day = date
            if i < (len(sorted_total_dates) - 1):
                end_day = pd.to_datetime(sorted_total_dates[i+1], format='%d-%m-%Y')
                try:
                    end_day_index = initial_port_df.index.get_loc(end_day) - 1
                    end_day_new = initial_port_df.index[end_day_index]
                except:
                    end_day_new = end_day
            else:
                end_day_new = initial_port_df.index[-1]
            
            # Calculate rebalancing parameters
            date_index = initial_port_df.index.get_loc(start_day)
            start_index = freq - (date_index % freq)
            
            # Get period data
            period_port_df = initial_port_df.loc[start_day:end_day_new]
            period_leverage_df = df_multiplied.loc[start_day:end_day_new]
            period_df_cash = initial_df_cash.loc[start_day:end_day_new]
            
            # Perform rebalancing
            rebalanced_portfolio, cash_rebalanced = rebalancing2(
                period_port_df, period_leverage_df, new_weights, freq, 
                invest_amount, start_index, cash_percent, total_invest_amount, 
                period_df_cash, cash_amount_end)
            
            # Update values for next iteration
            invest_amount = rebalanced_portfolio.iloc[-1].sum()
            cash_amount_end = cash_rebalanced.iloc[-1].item()
            total_invest_amount = invest_amount + cash_amount_end
            
            # Update DataFrames
            initial_port_df.loc[start_day:end_day_new] = rebalanced_portfolio
            initial_df_cash.loc[start_day:end_day_new] = cash_rebalanced

        # Calculate final portfolio value
        df_port_sum = initial_port_df.sum(axis=1)
        df_port_sum = df_port_sum.mask(df_port_sum.eq(0)).ffill()
        df_sum_final = df_port_sum + initial_df_cash['Cash']
        df_final = df_sum_final.to_frame(name='Portfolio Value')

        return df_final, initial_port_df

    except Exception as e:
        logging.error(f"Error in weights_rebalance_frequency: {str(e)}")
        raise
      
def weights_rebalancing_main(df, weights, invest_amount, df_multiplied, initial_df_cash):
    """
    Main function for portfolio rebalancing with weights adjustment.
    """
    try:
        # Validate inputs
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")
        if invest_amount <= 0:
            raise ValueError("Investment amount must be positive")

        # Create working copy and convert to float
        initial_port_df = df.copy().astype(float)
        logging.info(f"Starting portfolio rebalancing with {len(initial_port_df)} rows")

        # Get sorted dates for rebalancing
        sorted_total_dates, stock_dates = date_dict(initial_port_df)
        if not sorted_total_dates:
            raise ValueError("No valid dates for rebalancing")

        # Process each rebalancing period
        for i, date in enumerate(sorted_total_dates):
            # Get new weights based on availability
            new_weights_bool = bool_list(date, initial_port_df, stock_dates)
            new_weights = rebalance_weights(weights, new_weights_bool)

            # Calculate period dates
            start_day = sorted_total_dates[i]
            if i < (len(sorted_total_dates) - 1):
                end_day = sorted_total_dates[i + 1]
                end_day_index = df.index.get_loc(end_day) - 1
                end_day_new = df.index[end_day_index]
            else:
                end_day_new = None

            # Extract period data
            period_slice = slice(start_day, end_day_new)
            period_port_df = df.loc[period_slice]
            period_leverage_df = df_multiplied.loc[period_slice]

            # Calculate investment amounts and rebalance
            invest_amounts = [invest_amount * w for w in new_weights]
            rebalanced_portfolio = multi_portfolio(
                period_port_df, 
                period_leverage_df, 
                invest_amounts
            )

            # Update investment amount for next period
            invest_amount = rebalanced_portfolio.iloc[-1].sum()

            # Update portfolio values
            initial_port_df.loc[period_slice] = rebalanced_portfolio

            logging.info(f"Completed rebalancing for period {i+1}/{len(sorted_total_dates)}")

        # Calculate final portfolio values
        df_port_sum = initial_port_df.sum(axis=1)
        df_sum_final = df_port_sum + initial_df_cash['Cash']
        df_final = df_sum_final.to_frame(name='Portfolio Value')

        logging.info("Portfolio rebalancing completed successfully")
        return df_final, initial_port_df

    except Exception as e:
        logging.error(f"Error in weights_rebalancing_main: {str(e)}")
        raise

def sort_dates(dates_list):
    """
    Sort dates in chronological order.
    """
    try:
        return sorted(dates_list, key=lambda x: datetime.strptime(x, '%d-%m-%Y'))
    except Exception as e:
        logging.error(f"Error sorting dates: {str(e)}")
        raise

def get_rebalance_indexlist(df, freq): # function to get index at which rebalancing needs to happen

    df_index = df.reset_index().rename(columns={"index":"date"}) #resetting index

    #start_index = df_index.loc[df_index['date'] == rebalance_date].index[0]

    start_index = 0

    end_index = df_index.index[-1]

    #rebalance_date_pd = pd.to_datetime(rebalance_date, format='%d-%m-%Y') # converting date to pandas datetime

    #end_date = port_df.index[-1]

    index_list = list(range(start_index, (end_index + 1) , freq))


    return index_list

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
    df_index = initial_port_df.reset_index().rename(columns={"index": "date"})  # resetting index
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

    df_index = initial_port_df.reset_index().rename(columns={"index":"date"}) #resetting index


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
