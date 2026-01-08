import pandas_datareader.data as web # https://pandas-datareader.readthedocs.io/en/latest/index.html. same as from pandas_datareader import data but were replacing data with web
import datetime as dt
import yfinance as yf #wrapper for yahoo finance data. yfinance documentation https://ranaroussi.github.io/yfinance/reference/index.html

# Using stooq
df = web.DataReader("GE", "stooq", start="2024-09-10", end="2024-10-09") # Fetches the raw data into a pandas dataFrame
print(df.sort_index(ascending=False))  # Can manipulate the Dataframe using pandas; pandas_datareader automatically imports it for us

# Using yahoo finance
# start = dt.datetime(2020, 1, 1)
# end = dt.datetime(2020, 3, 30)
# yahoo = yf.download("GE", start, end, progress=False)
# print(yahoo)