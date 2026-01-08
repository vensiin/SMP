import pandas as pd
import lxml
import os
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf #wrapper for yahoo finance data. yfinance documentation https://ranaroussi.github.io/yfinance/reference/index.html

# Reads an xml file
# df = pandas.read_xml("C:/LTM SMP/Stock market predictor/test.xml", xpath="//note")
# # xpath: https://www.w3schools.com/xml/xpath_syntax.asp
# print(df)

df = web.DataReader("GE", "stooq", start="2024-09-10", end="2024-10-09")
print(df.sort_index().head())




