import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "source_raw.csv")
OUTPUT_A = os.path.join(BASE_DIR, "source_a.csv")
OUTPUT_B = os.path.join(BASE_DIR, "source_b.csv")

# Source A subset
SOURCE_A_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP",
    "ADSK", "AEP", "AMAT", "AMD", "AMGN", "^NDX"
]

# Step 1: Read skipping header lines
df_raw = pd.read_csv(INPUT_FILE, skiprows=2, header=None)

records = []

for _, row in df_raw.iterrows():
    date = pd.to_datetime(row[0])

    # Skip rows with no data
    non_null = row.dropna()[1:]  # drop date
    if len(non_null) < 6:
        continue

    values = non_null.values
    ticker = values[0]
    open_, high, low, close, volume = values[1:6]

    records.append({
        "Date": date,
        "Ticker": ticker,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

# Final long-format DataFrame
df_long = pd.DataFrame(records)

# Save Source A
df_a = df_long[df_long["Ticker"].isin(SOURCE_A_TICKERS)]
df_a.to_csv(OUTPUT_A, index=False)
print(f"✅ Saved Source A with {len(df_a)} rows to {OUTPUT_A}")

# Save Source B (everything)
df_long.to_csv(OUTPUT_B, index=False)
print(f"✅ Saved Source B with {len(df_long)} rows to {OUTPUT_B}")
