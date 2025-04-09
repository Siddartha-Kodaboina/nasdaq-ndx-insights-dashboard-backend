import pandas as pd
import numpy as np

# Create a sample dataframe with dates
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6],
    'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL']
}, index=pd.date_range('2023-01-01', periods=6, freq='D'))

print('Original DataFrame:')
print(df)

# Test different frequency strings
frequencies = ['D', 'W', 'M', 'ME', 'Q', 'QE', 'Y', 'YE']

for freq in frequencies:
    try:
        resampled = df['value'].resample(freq).last()
        print(f'\nResampling with {freq} - SUCCESS:')
        print(resampled)
    except Exception as e:
        print(f'\nResampling with {freq} - ERROR: {e}')
