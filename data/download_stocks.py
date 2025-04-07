import yfinance as yf
import pandas as pd
import os

# List of tickers (replace with actual tickers from your chosen index)
tickers = [
    "AAPL",  # 1
    "ABNB",  # 2
    "ADBE",  # 3
    "ADI",  # 4
    "ADP",  # 5
    "ADSK",  # 6
    "AEP",  # 7
    "AMAT",  # 8
    "AMD",  # 9
    "AMGN",  # 10
    "^NDX", # 102
    "AMZN",  # 11
    "ANSS",  # 12
    "APP",  # 13
    "ARM",  # 14
    "ASML",  # 15
    "AVGO",  # 16
    "AXON",  # 17
    "AZN",  # 18
    "BIIB",  # 19
    "BKNG",  # 20
    "BKR",  # 21
    "CCEP",  # 22
    "CDNS",  # 23
    "CDW",  # 24
    "CEG",  # 25
    "CHTR",  # 26
    "CMCSA",  # 27
    "COST",  # 28
    "CPRT",  # 29
    "CRWD",  # 30
    "CSCO",  # 31
    "CSGP",  # 32
    "CSX",  # 33
    "CTAS",  # 34
    "CTSH",  # 35
    "DASH",  # 36
    "DDOG",  # 37
    "DXCM",  # 38
    "EA",  # 39
    "EXC",  # 40
    "FANG",  # 41
    "FAST",  # 42
    "FTNT",  # 43
    "GEHC",  # 44
    "GFS",  # 45
    "GILD",  # 46
    "GOOG",  # 47
    "GOOGL",  # 48
    "HON",  # 49
    "IDXX",  # 50
    "INTC",  # 51
    "INTU",  # 52
    "ISRG",  # 53
    "KDP",  # 54
    "KHC",  # 55
    "KLAC",  # 56
    "LIN",  # 57
    "LRCX",  # 58
    "LULU",  # 59
    "MAR",  # 60
    "MCHP",  # 61
    "MDB",  # 62
    "MDLZ",  # 63
    "MELI",  # 64
    "META",  # 65
    "MNST",  # 66
    "MRVL",  # 67
    "MSFT",  # 68
    "MSTR",  # 69
    "MU",  # 70
    "NFLX",  # 71
    "NVDA",  # 72
    "NXPI",  # 73
    "ODFL",  # 74
    "ON",  # 75
    "ORLY",  # 76
    "PANW",  # 77
    "PAYX",  # 78
    "PCAR",  # 79
    "PDD",  # 80
    "PEP",  # 81
    "PLTR",  # 82
    "PYPL",  # 83
    "QCOM",  # 84
    "REGN",  # 85
    "ROP",  # 86
    "ROST",  # 87
    "SBUX",  # 88
    "SNPS",  # 89
    "TEAM",  # 90
    "TMUS",  # 91
    "TSLA",  # 92
    "TTD",  # 93
    "TTWO",  # 94
    "TXN",  # 95
    "VRSK",  # 96
    "VRTX",  # 97
    "WBD",  # 98
    "WDAY",  # 99
    "XEL",  # 100
    "ZS",  # 101 (just in case we need one extra for validation)

    "^IXIC" # 103
]

# Set date range
start_date = "2015-04-04"
end_date = "2025-04-05"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_RAW_DATA = os.path.join(BASE_DIR, "source_raw.csv")
# Initialize an empty list to store data
data_list = []
source_a_tickers = tickers[:11]

# Download data for all tickers and append to the list
for idx, ticker in enumerate(tickers):
    print(f"📥 Downloading: {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date)
    df["Ticker"] = ticker
    df.reset_index(inplace=True)  # Normalize
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]  # Keep clean cols
    data_list.append(df)

    # # Save Source A after first 10 stocks + ^NDX
    # if idx == len(source_a_tickers) - 1:
    #     source_a_df = pd.concat(data_list)
    #     source_a_df.to_csv(SOURCE_A, index=False)
    #     print("✅ Source A saved as source_a_ndx.csv")

# Save Source B (everything)
source_b_df = pd.concat(data_list)
source_b_df.to_csv(SOURCE_RAW_DATA, index=False)
print("✅ Source B saved as source_b_all.csv")
