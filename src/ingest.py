import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SHEET_URL = os.getenv("SHEET_URL")
SEED_PATH = os.getenv("SEED_PATH", "data/seed/Marketing_data.csv")

def fetch_live():
    if SHEET_URL:
        try:
            df = pd.read_csv(SHEET_URL)
            source = "google_sheet"
        except Exception as e:
            print(f"[warn] Failed to read SHEET_URL: {e}. Falling back to seed CSV.")
            df = pd.read_csv(SEED_PATH)
            source = "seed_csv_fallback"
    else:
        df = pd.read_csv(SEED_PATH)
        source = "seed_csv"
    # Expected columns (align with your existing project)
    expected = [
        "CUST_ID","BALANCE","BALANCE_FREQUENCY","PURCHASES","ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY","CASH_ADVANCE_TRX","PURCHASES_TRX",
        "CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PRC_FULL_PAYMENT","TENURE"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    df = df[expected].copy()
    df["_ingested_at"] = datetime.utcnow()
    df["_source"] = source
    return df

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = fetch_live()
    out = "data/processed/live_raw.csv"
    df.to_csv(out, index=False)
    print(f"[ingest] wrote {len(df)} rows to {out}")
