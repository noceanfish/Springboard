# -*- coding: utf-8 -*-
# stk_predictor/visualization/visualize.py
#

import matplotlib.pyplot as plt
import datetime

from flask import current_app
import os


def plot_history_price(ticker, df):
    """plot the history price line
    Parameters
    ----------
    ticker: str
        the name of prediction company
    df: DataFrame
        the historical trading info, including close price, volumes,

    Return
    ----------
    png_path: str
        return the graph path
    """
    fig = plt.figure(figsize=(16, 6.5))
    plot_name = ticker.upper() + '_' + datetime.datetime.today().strftime('%y-%m-%d_%H-%M-%S') + '.png'
    plt.plot(df['intraday_close'], label=ticker)
    plt.title('Adjust Closing Price')
    plt.savefig(os.path.join(current_app.root_path, 'static/visualization', plot_name))
    plt.close(fig)
    return plot_name