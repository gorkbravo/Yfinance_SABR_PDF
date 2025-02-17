# main.py

from Data_Collection_Engine import data_collection_main
from Data_Cleaning_Engine import clean_options_data
from Bflys_engine import bfly_main
from SABR_betas_comp_engine import main as sabr_betas_comp_main
from SABR_engine import main as sabr_main

def main_pipeline():
    print("----- Start Pipeline -----\n")

    # 1) Prompt user for ticker/days
    ticker = input("Enter ticker (e.g. 'SPY'): ").upper().strip()
    days_str = input("Enter approx number of days to expiration (e.g. '30'): ").strip()

    try:
        target_days = int(days_str)
    except ValueError:
        target_days = 30

    # 2) Data Collection
    print("\n1) Running Data Collection Engine...")
    csv_file_collected = data_collection_main(ticker, target_days)
    if csv_file_collected is None:
        print("[ERROR] Data Collection failed. Exiting pipeline.")
        return

    # 3) Data Cleaning
    cleaned_file = csv_file_collected.replace(".csv", "_cleaned.csv")
    print(f"\n2) Running Data Cleaning Engine on {csv_file_collected} ...")
    clean_options_data(csv_file_collected, cleaned_file)

    # 4) Butterfly Engine (pass the cleaned file)
    print("\n3) Running Butterfly (Bfly) Engine...")
    bfly_main(cleaned_file)

    # 5) SABR Betas Comparison
    print("\n4) Running SABR Betas Comparison Engine...")
    F = float(input("Enter the value of F (spot/forward value): ").strip())
    T = float(input("Enter the value of T (time to expiration in years): ").strip())
    sabr_betas_comp_main(cleaned_file, F, T)
        # 1/52 = 0.01923076923      1 week
        # 4/52 = 0.07692307692      1 month (approx)
        # 8/52 = 0.1538461538       
        # 12/52 = 0.2307692308      
        # 16/52 = 0.3076923077      
        # 20/52 = 0.3846153846      
        # 24/52 = 0.4615384615      
        # 28/52 = 0.5384615385      
        # 32/52 = 0.6153846154      
        # 36/52 = 0.6923076923      
        # 40/52 = 0.7692307692      
        # 44/52 = 0.8461538462        
        # 48/52 = 0.9230769231
        # 52/52 = 1                 1 year (approx)

    # 6) SABR Engine
    print("\n5) Running SABR Engine...")
    beta = float(input("Enter the beta value to use in SABR engine: ").strip())
    sabr_main(cleaned_file, F, T, beta)

    print("\n----- Pipeline Complete! -----")

if __name__ == "__main__":
    main_pipeline()


