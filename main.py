import pandas as pd
stocks_data = pd.read_csv("stocks.csv")
print(stocks_data.head())
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
missing_values = stocks_data.isnull().sum()
unique_stocks = stocks_data['Ticker'].value_counts()

time_range = stocks_data['Date'].min(), stocks_data['Date'].max()
print(time_range)

import matplotlib.pyplot as plt


plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8))
for ticker in unique_stocks.index:
    subset = stocks_data[stocks_data['Ticker'] == ticker]
    ax.plot(subset['Date'], subset['Close'], label=ticker)

ax.set_title('Closing Price Trends for AAPL, MSFT, NFLX, GOOG')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price(USD)')
ax.legend()

plt.xticks(rotation=45)
plt.show()

from statsmodels.tsa.stattools import adfuller
def adf_test(series, title=''):
    print(f'ADF Test on"{title}')
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations used']
    out = pd.Series(result[0:4], index=labels)

    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    print("\n")

for ticker in unique_stocks.index:
    series = stocks_data[stocks_data['Ticker'] == ticker]['Close']
    adf_test(series, title=ticker)

    stocks_data["Diff-Close"] = stocks_data.groupby('Ticker')['Close'].transform(lambda x: x.diff())

for ticker in unique_stocks.index:
    series = stocks_data[stocks_data['Ticker'] == ticker]['Diff-Close'].dropna()
    adf_test(series, title=f"{ticker} - Differenced")

from statsmodels.tsa.vector_ar.var_model import VAR
var_data = stocks_data.pivot(index='Date', columns='Ticker', values='Diff-Close').dropna()

model = VAR(var_data)
model_fitted = model.fit(ic='aic')
forecast_steps = 5
forecasted_values = model_fitted.forecast(var_data.values[-model_fitted.k_ar:], steps=forecast_steps)
forecasted_df = pd.DataFrame(forecasted_values, index=pd.date_range(start=var_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'),columns=var_data.columns)

for column in forecasted_df.columns:
    forecasted_df[column] = (stocks_data.groupby('Ticker')['Close'].last()[column] + forecasted_df[column].cumsum())

print(forecasted_df)

fig, ax = plt.subplots(figsize=(14, 8))
for ticker in unique_stocks.index:
    historical_data = stocks_data[stocks_data['Ticker'] == ticker]
    ax.plot(historical_data['Date'], historical_data['Close'], label=ticker)

for column in forecasted_df.columns:
    ax.plot(forecasted_df.index, forecasted_df[column], label=f'{column} Forecast', linestyle='--')

ax.set_title('Historical and Forecasted Closing Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price(USD)')
ax.legend()

plt.xticks(rotation=45)
plt.show()
