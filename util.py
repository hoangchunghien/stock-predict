import pandas as pd
import os
import matplotlib.pyplot as plt


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), delimiter='|', index_col='Date',
                              parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp, how='inner')

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_rolling_mean(values, window):
    return pd.rolling_mean(values, window=window)


def normalize_data(df):
    return df / df.ix[0, :]


def get_rolling_std(values, window):
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band


def compute_macd(df):
    ema1 = df.ewm(span=12).mean()
    ema2 = df.ewm(span=26).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    return macd, signal, histogram


def compute_stochastic_oscillator(values, window):
    l, h = values.rolling(window=window).min(), values.rolling(window=window).max()
    k = 100 * (values - l) / (h - l)
    d = k.rolling(window=3).mean()
    return k, d


def compute_rsi(values, n):
    dup, ddown = values.copy(), values.copy()
    dup[dup < 0] = 0
    ddown[ddown > 0] = 0
    rolup = dup.rolling(n).mean()
    roldown = ddown.rolling(n).mean().abs()
    RS = rolup / roldown
    return 100 - 100 / (1 + RS)


def compute_daily_returns(df):
    daily_return = (df / df.shift(1)) - 1
    daily_return.ix[0, :] = 0
    return daily_return


def compute_cumulative_returns(df):
    cmr = (df / df.ix[0, :]) - 1
    return cmr


def symbol_to_path(symbol, base_dir="HOSE"):
    """Return CSV file path given ticker symbol"""
    return os.path.join(base_dir, "{}.CSV".format(symbol))
