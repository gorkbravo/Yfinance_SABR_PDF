import yfinance as yf
import pandas as pd
import datetime as dt
import os

def download_options_data(ticker: str, target_days: int):
    """
    Download the options chain of `ticker` that expires ~`target_days` from today.
    Returns a combined DataFrame, plus calls DataFrame, puts DataFrame, and the actual expiration string used.
    """
    # Create a Ticker object
    ticker_obj = yf.Ticker(ticker)

    # Get all available option expiration dates
    all_expirations = ticker_obj.options
    if not all_expirations:
        raise ValueError(f"No expiration dates found for ticker '{ticker}'")

    today = dt.date.today()
    target_date = today + dt.timedelta(days=target_days)

    best_exp = None
    min_diff = dt.timedelta(days=9999999)

    for exp_str in all_expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        # Only consider future expirations
        if exp_date > today:
            diff = abs(exp_date - target_date)
            if diff < min_diff:
                min_diff = diff
                best_exp = exp_date

    if best_exp is None:
        raise ValueError(f"No valid (future) expiration found for ~{target_days} days out.")

    best_exp_str = best_exp.strftime("%Y-%m-%d")
    print(f"[INFO] Closest expiration to {target_days} days: {best_exp_str} for {ticker}")

    # Download option chain for that expiration
    chain = ticker_obj.option_chain(best_exp_str)
    calls, puts = chain.calls, chain.puts

    # Add a 'type' column to each
    calls["type"] = "call"
    puts["type"] = "put"

    # Combine calls/puts
    options_data = pd.concat([calls, puts], ignore_index=True)
    return options_data, calls, puts, best_exp_str


def data_collection_main(ticker: str, target_days: int):
    """
    Uses the user-provided `ticker` and `target_days` to download and save options data.
    Returns the CSV path so the pipeline can continue if needed.
    """
    output_folder = "C:/Users/User/Desktop/UPF/TGF/Data"
    
    # 1. Download the data
    try:
        options_data, calls, puts, exp_used = download_options_data(ticker, target_days)
    except Exception as e:
        print(f"[ERROR] Could not download options data: {e}")
        return None

    # 2. Build filenames based on ticker and days
    # e.g. "SPY_opt_30days.csv" and "SPY_opt_30days.xlsx"
    csv_filename = f"{ticker}_opt_{target_days}days.csv"
    # excel_filename = f"{ticker}_opt_{target_days}days.xlsx"

    csv_path = os.path.join(output_folder, csv_filename)
    # excel_path = os.path.join(output_folder, excel_filename)

    # 3. Save to CSV
    try:
        options_data.to_csv(csv_path, index=False)
        print(f"[INFO] Saved options data to {csv_path}")
        print(f"[INFO] Expiration used: {exp_used}")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")

    # 4. Save to Excel (two sheets)
    # try:
    #     with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    #         calls.to_excel(writer, sheet_name="Calls", index=False)
    #         puts.to_excel(writer, sheet_name="Puts", index=False)
    #     print(f"[INFO] Saved options data to Excel: {excel_path}")
    # except Exception as e:
    #     print(f"[ERROR] Failed to save Excel: {e}")

    # Return CSV path if the next step in your pipeline needs it
    return csv_path

# If you want to test this file alone, you could uncomment this:
#
# if __name__ == "__main__":
#     test_ticker = "SPY"
#     test_days = 30
#     data_collection_main(test_ticker, test_days)
