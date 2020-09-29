# -*- coding: utf-8 -*-
"""Click commands."""

import os
import pandas as pd
import pickle
import datetime

import click
from flask import current_app, Blueprint

from .data import make_dataset
from .models.train_sa_model import SentimentalAnalysisModel
from .models.train_ts_model import TimeSeriesModel
from .extensions import db

HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.join(HERE, os.pardir)
TEST_PATH = os.path.join(PROJECT_ROOT, "tests")

blueprint = Blueprint('cli_command', __name__, cli_group=None)


@blueprint.cli.command("test")
def test():
    """RUN the tests."""
    import pytest

    rv = pytest.main([TEST_PATH, "--verbose"])
    exit(rv)


@blueprint.cli.command("train_sa_model")
def train_sentimental_model():
    # get phasebank dataset from local filesystem
    file_path_ph100 = "data/raw/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
    file_path_ph75 = "data/raw/FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
    df_100 = make_dataset.get_raw_phasebank_dataset(
        os.path.join(os.path.dirname(HERE), file_path_ph100))

    df_75 = make_dataset.get_raw_phasebank_dataset(
        os.path.join(os.path.dirname(HERE), file_path_ph75))

    df = pd.concat([df_100, df_75])
    df.target = df.target.astype('int32')
    current_app.logger.info("get raw dataset from local filesystem. {}".format(df.info()))

    sa_model = SentimentalAnalysisModel()
    cleaned_df = sa_model.dataset_preprocessing(df.text.values)
    x_train, x_test, y_train, y_test = sa_model.dataset_split(cleaned_df, df.target)

    # generate elmo embeddings
    x_train_elmo = sa_model.generate_elmo_embeddings(x_train)

    x_test_elmo = sa_model.generate_elmo_embeddings(x_test)

    classifier = sa_model.run_logistic_regression(x_train_elmo, x_test_elmo, y_train, y_test)
    with open("models/sa_classifier.pkl", 'wb') as f:
        pickle.dump(classifier, f)
    current_app.logger.info("Successful train sentimental model.")


@blueprint.cli.command("train_ts_model")
@click.argument("ticker")
def train_time_series_model(ticker='AAPL'):
    """Train the time-series model
    default ticker name is 'AAPL'
    """
    recent_trading_date = pd.read_sql('SELECT MAX(trading_date) FROM AAPL',
                                      db.engine,
                                      columns=['trading_date']).values[0, 0]
    if recent_trading_date is not None:
        start = datetime.datetime.strptime(recent_trading_date.split()[0], '%Y-%m-%d') + datetime.timedelta(1)
    else:
        start = datetime.datetime(1981, 1, 1)
    end = datetime.datetime.today()
    df = make_dataset.get_ticker_from_yahoo(ticker, start, end)
    
    ts_model = TimeSeriesModel(LSTM_units=220,
                               lookback=1,
                               dropout_rate=0.,
                               recurrent_dropout=0.4,
                               batch_size=1,
                               epochs=1,
                               scoring='accuracy',
                               verbose=1)
    
    features_df = ts_model.dataset_preprocessing(df)
    
    # # # test only
    # # features_df.to_csv('features_df.csv')

    # features_df = pd.read_csv('features_df.csv', index_col='Date')
    # # finished

    data = features_df.drop(['label'], axis=1)
    target = features_df['label']
    ts_classifier = ts_model.train_model_with_lstm(data, target)

    current_app.logger.info("Successful train time series model.")
    

@blueprint.cli.command("init-db")
def init_db_command():
    """Initilize DB through flask-sqlalchemy
    """
    current_app.logger.info("start initilize sqlite DB...")
    current_app.logger.info(current_app.config['SQLALCHEMY_DATABASE_URI'])
    try:
        from stk_predictor.predictor.models import Apple
        db.drop_all()
        db.create_all()
        # Apple.create(ids=1, trading_date=datetime.datetime(2020, 8, 29), intraday_close=109.1, intraday_volumes=20000.0)
        current_app.logger.info(db.session.query(Apple).scalar())
        current_app.logger.info("Successfully initilize sqlite DB!")
    except Exception as ex:
        current_app.logger.info("Initialize DB encounter exception.", ex)
        # raise RuntimeError("run time exception from init-db")

