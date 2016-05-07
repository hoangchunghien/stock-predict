import pandas as pd
import matplotlib.pyplot as plt
from util import *
import numpy as np


def test_run():
    start_date = '2015-12-01'
    end_date = '2015-12-31'
    dates = pd.date_range(start_date, end_date)
    symbols = ['SKG']
    df = get_data(symbols, dates)
    cmr = compute_cumulative_returns(df)
    ax = cmr.plot()
    rm = get_rolling_mean(cmr, 5)
    rm.plot(ax=ax)
    plt.show()


if __name__ == '__main__':
    test_run()
