# Data_Cleaning_Engine_Test.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def clean_options_data(input_file, output_file):
    """
    A more relaxed cleaning process that:
      1) Drops only rows with impliedVolatility < 1e-3,
      2) Removes rows where bid or ask is NaN or negative,
      3) (Optional) Removes zero volume if desired (commented out by default),
      4) Drops some unnecessary columns,
      5) Plots calls vs. puts IV smile,
      6) Saves the final cleaned file to CSV.
    """

    # 1. Load the data
    try:
        options_data = pd.read_csv(input_file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # Check for key columns
    has_iv = 'impliedVolatility' in options_data.columns
    has_strike = 'strike' in options_data.columns
    has_type = 'type' in options_data.columns  # "call" or "put"

    # 2. Remove near-zero IV
    if has_iv:
        iv_threshold = 1e-3
        size_before = len(options_data)
        options_data = options_data[options_data['impliedVolatility'] > iv_threshold]
        size_after = len(options_data)
        print(f"Dropped {size_before - size_after} rows with impliedVolatility <= {iv_threshold}.")

    # 3. Remove rows with missing or negative bid/ask
    #    (If you'd rather allow zero, change > 0 to >= 0)
    options_data = options_data.dropna(subset=['bid', 'ask'])
    size_before = len(options_data)
    options_data = options_data[(options_data['bid'] >= 0) & (options_data['ask'] >= 0)]
    size_after = len(options_data)
    print(f"Dropped {size_before - size_after} rows due to negative bid/ask or NaNs.")

    # 4. (Optional) Volume filter - comment out if you want to keep zero-volume options
    if 'volume' in options_data.columns:
        size_before = len(options_data)
        options_data = options_data[options_data['volume'] >= 0]
        size_after = len(options_data)
        print(f"Dropped {size_before - size_after} rows due to negative volume.")

    # 5. Drop unnecessary columns
    columns_to_remove = ['inTheMoney', 'contractSize', 'currency']
    options_data = options_data.drop(columns=columns_to_remove, errors='ignore')

    # 6. Extract expiration date from contractSymbol (if present)
    if 'contractSymbol' in options_data.columns:
        options_data['expirationDate'] = options_data['contractSymbol'].str.extract(r'(\d{6})')[0]
        options_data['expirationDate'] = pd.to_datetime(
            options_data['expirationDate'],
            format='%y%m%d',
            errors='coerce'
        )

    # 7. Calculate midprice (optional)
    options_data['midprice'] = (options_data['bid'] + options_data['ask']) / 2

    # 8. Smooth midprice (optional)
    options_data['smoothed_midprice'] = gaussian_filter1d(
        options_data['midprice'].fillna(0), sigma=3
    )

    # 9. Plot calls and puts if columns exist
    if has_iv and has_strike and has_type:
        calls_data = options_data[options_data['type'].str.lower() == 'call']
        puts_data  = options_data[options_data['type'].str.lower() == 'put']

        plt.figure(figsize=(10, 5))
        plt.scatter(
            calls_data['strike'],
            calls_data['impliedVolatility'],
            s=10, alpha=0.5, color='blue', label='Calls'
        )
        plt.scatter(
            puts_data['strike'],
            puts_data['impliedVolatility'],
            s=10, alpha=0.5, color='red', label='Puts'
        )
        plt.title("Implied Volatility vs. Strike ")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.show()

    # 10. Save cleaned data to CSV
    try:
        options_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the cleaned data: {e}")

    # 11. Check missing values
    missing_values = options_data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # 12. Print basic stats on expirationDate (optional)
    if 'expirationDate' in options_data.columns:
        print(options_data['expirationDate'].describe())

    # (Optional) Print additional info about strikes
    if has_strike:
        print("Min strike:", options_data['strike'].min())
        print("Max strike:", options_data['strike'].max())


# Optional: test block for running this file by itself
if __name__ == "__main__":
    test_input = "Add testing file"
    test_output = test_input.replace(".csv", "_cleaned.csv")
    clean_options_data(test_input, test_output)

