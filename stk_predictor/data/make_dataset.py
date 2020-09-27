# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
from pandas_datareader import data as pdr

from flask import current_app
from stk_predictor.extensions import db


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def get_ticker_from_yahoo(ticker, start_date, end_date):
    yf.pdr_override()
    try:
        new_trading_df = pdr.get_data_yahoo(
            ticker, start_date, end_date, interval='1d')
        new_trading_df = new_trading_df.drop(
            ['Open', 'High', 'Low', 'Adj Close'], axis=1)
        new_trading_df = new_trading_df.dropna('index')
        new_trading_df = new_trading_df.reset_index()
        new_trading_df.columns = ['trading_date',
                                  'intraday_close', 'intraday_volumes']

        his_trading_df = pd.read_sql('aapl', db.engine, index_col='id')
        df = pd.concat([his_trading_df, new_trading_df]
                       ).drop_duplicates('trading_date')
        df = df.sort_values(by='trading_date')
        df = df.reset_index(drop=True)

        if len(df) > 0:
            df.to_sql("aapl", db.engine, if_exists='replace', index_label='id')
            return df
        else:
            # t = pd.read_sql('aapl', db.engine, index_col='id')
            return None
    except Exception as ex:
        raise RuntimeError(
            "Catch Excetion when retrieve data from Yahoo...", ex)
        return None


def get_news_from_finviz(ticker):
    """Request news headline from finviz, according to 
    company ticker's name

    Parameters
    -----------
    ticker: str
        the stock ticker name

    Return
    ----------
    df : pd.DataFrame
        return the latest 2 days news healines.
    """
    current_app.logger.info("Job >> Enter Finviz news scrape step...")

    base_url = 'https://finviz.com/quote.ashx?t={}'.format(ticker)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) \
                       AppleWebKit/537.36 (KHTML, like Gecko) \
                       Chrome/50.0.2661.102 Safari/537.36'
    }
    parsed_news = []

    try:
        res = requests.get(base_url, headers=headers)
        if res.status_code == 200:
            texts = res.text
            soup = BeautifulSoup(texts)
            news_tables = soup.find(id="news-table")

            for x in news_tables.findAll('tr'):
                text = x.a.get_text()
                date_scrape = x.td.text.split()
                if len(date_scrape) == 1:
                    time = date_scrape[0]
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]

                parsed_news.append([date, time, text])

            # filter the recent day news
            df = pd.DataFrame(parsed_news, columns=['date', 'time', 'texts'])
            df['date'] = pd.to_datetime(df.date).dt.date
            one_day_period = (datetime.datetime.today() -
                              datetime.timedelta(days=1)).date()
            df_sub = df[df.date >= one_day_period]
            return df_sub

        else:
            raise RuntimeError("HTTP response Error {}".format(
                res.status_code)) from None
    except Exception as ex:
        current_app.logger.info("Exception in scrape Finviz.", ex)
        raise RuntimeError("Exception in scrape Finviz.") from ex


def prepare_trading_dataset(df):
    """Prepare the trading data set.
    Time series analysis incoporate previous data for future prediction,
    We need to retrieve historical data to generate features. 

    Parameters
    -----------
    df: DataFrame
        the stock ticker trading data, including trading-date, close-price, volumes
    window: int, default = 400
         feature engineer windows size. Using at most 400 trading days to construct
         features.

    Return
    ----------
    array_lstm : np.array
        return the array with 3 dimensions shape -> [samples, 1, features]
    """
    if len(df) == 0:
        raise RuntimeError(
            "Encounter Error in >>make_dataset.prepare_trading_dataset<<... \
            Did not catch any news.") from None
    else:
        df['log_ret_1d'] = np.log(df['intraday_close'] / df['intraday_close'].shift(1))

        df['log_ret_1w'] = pd.Series(df['log_ret_1d']).rolling(window=5).sum()
        df['log_ret_2w'] = pd.Series(df['log_ret_1d']).rolling(window=10).sum()
        df['log_ret_3w'] = pd.Series(df['log_ret_1d']).rolling(window=15).sum()
        df['log_ret_4w'] = pd.Series(df['log_ret_1d']).rolling(window=20).sum()
        df['log_ret_8w'] = pd.Series(df['log_ret_1d']).rolling(window=40).sum()
        df['log_ret_12w'] = pd.Series(df['log_ret_1d']).rolling(window=60).sum()
        df['log_ret_16w'] = pd.Series(df['log_ret_1d']).rolling(window=80).sum()
        df['log_ret_20w'] = pd.Series(df['log_ret_1d']).rolling(window=100).sum()
        df['log_ret_24w'] = pd.Series(df['log_ret_1d']).rolling(window=120).sum()
        df['log_ret_28w'] = pd.Series(df['log_ret_1d']).rolling(window=140).sum()
        df['log_ret_32w'] = pd.Series(df['log_ret_1d']).rolling(window=160).sum()
        df['log_ret_36w'] = pd.Series(df['log_ret_1d']).rolling(window=180).sum()
        df['log_ret_40w'] = pd.Series(df['log_ret_1d']).rolling(window=200).sum()
        df['log_ret_44w'] = pd.Series(df['log_ret_1d']).rolling(window=220).sum()
        df['log_ret_48w'] = pd.Series(df['log_ret_1d']).rolling(window=240).sum()
        df['log_ret_52w'] = pd.Series(df['log_ret_1d']).rolling(window=260).sum()
        df['log_ret_56w'] = pd.Series(df['log_ret_1d']).rolling(window=280).sum()
        df['log_ret_60w'] = pd.Series(df['log_ret_1d']).rolling(window=300).sum()
        df['log_ret_64w'] = pd.Series(df['log_ret_1d']).rolling(window=320).sum()
        df['log_ret_68w'] = pd.Series(df['log_ret_1d']).rolling(window=340).sum()
        df['log_ret_72w'] = pd.Series(df['log_ret_1d']).rolling(window=360).sum()
        df['log_ret_76w'] = pd.Series(df['log_ret_1d']).rolling(window=380).sum()
        df['log_ret_80w'] = pd.Series(df['log_ret_1d']).rolling(window=400).sum()

        df['vol_1w'] = pd.Series(df['log_ret_1d']).rolling(window=5).std()*np.sqrt(5)
        df['vol_2w'] = pd.Series(df['log_ret_1d']).rolling(window=10).std()*np.sqrt(10)
        df['vol_3w'] = pd.Series(df['log_ret_1d']).rolling(window=15).std()*np.sqrt(15)
        df['vol_4w'] = pd.Series(df['log_ret_1d']).rolling(window=20).std()*np.sqrt(20)
        df['vol_8w'] = pd.Series(df['log_ret_1d']).rolling(window=40).std()*np.sqrt(40)
        df['vol_12w'] = pd.Series(df['log_ret_1d']).rolling(window=60).std()*np.sqrt(60)
        df['vol_16w'] = pd.Series(df['log_ret_1d']).rolling(window=80).std()*np.sqrt(80)
        df['vol_20w'] = pd.Series(df['log_ret_1d']).rolling(window=100).std()*np.sqrt(100)
        df['vol_24w'] = pd.Series(df['log_ret_1d']).rolling(window=120).std()*np.sqrt(120)
        df['vol_28w'] = pd.Series(df['log_ret_1d']).rolling(window=140).std()*np.sqrt(140)
        df['vol_32w'] = pd.Series(df['log_ret_1d']).rolling(window=160).std()*np.sqrt(160)
        df['vol_36w'] = pd.Series(df['log_ret_1d']).rolling(window=180).std()*np.sqrt(180)
        df['vol_40w'] = pd.Series(df['log_ret_1d']).rolling(window=200).std()*np.sqrt(200)
        df['vol_44w'] = pd.Series(df['log_ret_1d']).rolling(window=220).std()*np.sqrt(220)
        df['vol_48w'] = pd.Series(df['log_ret_1d']).rolling(window=240).std()*np.sqrt(240)
        df['vol_52w'] = pd.Series(df['log_ret_1d']).rolling(window=260).std()*np.sqrt(260)
        df['vol_56w'] = pd.Series(df['log_ret_1d']).rolling(window=280).std()*np.sqrt(280)
        df['vol_60w'] = pd.Series(df['log_ret_1d']).rolling(window=300).std()*np.sqrt(300)
        df['vol_64w'] = pd.Series(df['log_ret_1d']).rolling(window=320).std()*np.sqrt(320)
        df['vol_68w'] = pd.Series(df['log_ret_1d']).rolling(window=340).std()*np.sqrt(340)
        df['vol_72w'] = pd.Series(df['log_ret_1d']).rolling(window=360).std()*np.sqrt(360)
        df['vol_76w'] = pd.Series(df['log_ret_1d']).rolling(window=380).std()*np.sqrt(380)
        df['vol_80w'] = pd.Series(df['log_ret_1d']).rolling(window=400).std()*np.sqrt(400)

        df['volume_1w'] = pd.Series(df['intraday_volumes']).rolling(window=5).mean()
        df['volume_2w'] = pd.Series(df['intraday_volumes']).rolling(window=10).mean()
        df['volume_3w'] = pd.Series(df['intraday_volumes']).rolling(window=15).mean()
        df['volume_4w'] = pd.Series(df['intraday_volumes']).rolling(window=20).mean()
        df['volume_8w'] = pd.Series(df['intraday_volumes']).rolling(window=40).mean()
        df['volume_12w'] = pd.Series(df['intraday_volumes']).rolling(window=60).mean()
        df['volume_16w'] = pd.Series(df['intraday_volumes']).rolling(window=80).mean()
        df['volume_20w'] = pd.Series(df['intraday_volumes']).rolling(window=100).mean()
        df['volume_24w'] = pd.Series(df['intraday_volumes']).rolling(window=120).mean()
        df['volume_28w'] = pd.Series(df['intraday_volumes']).rolling(window=140).mean()
        df['volume_32w'] = pd.Series(df['intraday_volumes']).rolling(window=160).mean()
        df['volume_36w'] = pd.Series(df['intraday_volumes']).rolling(window=180).mean()
        df['volume_40w'] = pd.Series(df['intraday_volumes']).rolling(window=200).mean()
        df['volume_44w'] = pd.Series(df['intraday_volumes']).rolling(window=220).mean()
        df['volume_48w'] = pd.Series(df['intraday_volumes']).rolling(window=240).mean()
        df['volume_52w'] = pd.Series(df['intraday_volumes']).rolling(window=260).mean()
        df['volume_56w'] = pd.Series(df['intraday_volumes']).rolling(window=280).mean()
        df['volume_60w'] = pd.Series(df['intraday_volumes']).rolling(window=300).mean()
        df['volume_64w'] = pd.Series(df['intraday_volumes']).rolling(window=320).mean()
        df['volume_68w'] = pd.Series(df['intraday_volumes']).rolling(window=340).mean()
        df['volume_72w'] = pd.Series(df['intraday_volumes']).rolling(window=360).mean()
        df['volume_76w'] = pd.Series(df['intraday_volumes']).rolling(window=380).mean()
        df['volume_80w'] = pd.Series(df['intraday_volumes']).rolling(window=400).mean()

        df = df.dropna(axis=0)
        df = df.drop(['trading_date', 'intraday_close', 'intraday_volumes', 'log_ret_1d'], axis=1)
        array_lstm = df.values.reshape(df.shape[0], 1, df.shape[1])
        return array_lstm

def get_raw_phasebank_dataset(file_path):
    """Retrieve the raw dataset from local-filesystem

    Parameters
    -----------
    ticker: str
        the stock ticker name

    Return
    ----------
    df : pd.DataFrame
        return the phasebank dataset.
    """
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    corpus = []
    target = []
    for line in lines:
        if len(line.split('@')) == 2:
            text = line.split('@')[0]
            corpus.append(text)
            label = line.split('@')[1].strip()
            target.append(label)

    current_app.logger.info('total lines: {}; the df_p100 shape is {}'.format(
        len(lines), (len(corpus), len(target))))
    df = pd.DataFrame(list(zip(corpus, target)), columns=['text', 'target'])
    df.loc[df['target'] == 'positive', 'target'] = 1
    df.loc[df['target'] == 'negative', 'target'] = -1
    df.loc[df['target'] == 'neutral', 'target'] = 0
    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
