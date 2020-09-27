# -*- coding: utf-8 -*-
# stk_predictor/model/train_ts_model.py
#

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from flask import current_app


# Define a callback class
# Resets the states after each epoch (after going through a full time series)
class ModelStateReset(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()


reset=ModelStateReset()


class TimeSeriesModel(object):
    """Generate time series model
    based on price, volumes, volatility and LSTM
    """

    def __init__(self, 
                 LSTM_units,
                 lookback,
                 dropout_rate,
                 recurrent_dropout,
                 batch_size,
                 epochs,
                 scoring,
                 verbose):
        self.LSTM_units = LSTM_units
        self.lookback = lookback
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = 1
        self.num_features = None
        self.scoring_lstm = scoring

        dev_size = 0.1
        n_splits = int((1 // dev_size) - 1)
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def dataset_preprocessing(self, df):
        """Preprocess the trading info Dataset
        Parameters
        ----------
        df: DataFrame
            the historical trading info, including close price, volumes,
        
        Return
        ----------
        df: DataFrame
            the processed dataset, contains:
            'trading_date', 'intraday_close', 'intraday_close', 'log_ret_1d'
            'log_ret_1w ~ log_ret_80w',
            'vol_1w ~ vol_80w'
            'volume_1w ~ volume_80w'
            totally 73 columns
        """
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

        df['return_label'] = pd.Series(df['log_ret_1d']).shift(-21).rolling(window=21).sum()
        df['label'] = np.where(df['return_label'] > 0, 1, 0)
        df = df.drop(['return_label'], axis=1)
        df = df.dropna(axis=0)

        current_app.logger.info(
            "finish dataset preprocessing. Totally get {} rows, {} columns".format(len(df), len(df.columns))
        )

        return df

    def create_shallow_lstm(self):
        model = Sequential()
    
        model.add(LSTM(units=self.LSTM_units, 
                       batch_input_shape=(self.num_samples, self.lookback, self.num_features),
                       stateful=True,
                       recurrent_dropout=self.recurrent_dropout))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.he_normal(seed=1)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model_with_lstm(self, data, target):
        """Train time-series model using LSTM
        Parameters
        ----------
        data: float
            the historical trading info, used for training model
        target: int
            the labels
        
        Return
        ----------
        ts_model: save in local file system
        """

        # ignore 'index:date' 'close', 'volume', 'log_ret_1d'
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 3:73],
                                                            target, 
                                                            test_size=0.1,
                                                            shuffle=False,
                                                            stratify=None)

        # standard the input data and reshap to lstm input
        pipeline = Pipeline([
            ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True))
        ])
        x_train_standard = pipeline.fit_transform(x_train)
        x_test_standard = pipeline.transform(x_test)
        x_train_standard_lstm = x_train_standard.reshape(x_train_standard.shape[0],
                                                         1,
                                                         x_train_standard.shape[1])
        x_test_standard_lstm = x_test_standard.reshape(x_test_standard.shape[0],
                                                       1,
                                                       x_test_standard.shape[1])

        self.num_features = x_train_standard.shape[1]

        # define classifier
        clr_lstm = KerasClassifier(build_fn=self.create_shallow_lstm,
                                   epochs=self.epochs)

        # define gridsearch
        hyperparameter = {'batch_size': [self.batch_size]}
        search_lstm = GridSearchCV(estimator=clr_lstm,
                                   param_grid=hyperparameter,
                                   n_jobs=None,
                                   cv=self.tscv, 
                                   scoring=self.scoring_lstm,  # accuracy
                                   refit=True, 
                                   return_train_score=False)

        tuned_model_lstm = search_lstm.fit(x_train_standard_lstm, y_train, shuffle=False, callbacks=[reset])

        y_pred = tuned_model_lstm.predict(x_test_standard_lstm)

        current_app.logger.info(
            'training accuracy:  {}'.format(tuned_model_lstm.best_score_))
        current_app.logger.info(
            'testing report: {}'.format(classification_report(y_test, y_pred)))

        # save the trained model
        tuned_model_lstm.best_estimator_.model.save('models/tuned_ts_model.h5')

        clr_lstm.fit(x_train_standard_lstm, y_train, shuffle=False, callbacks=[reset])
        clr_lstm.model.save('models/clr_model.h5')

        # test the save model and restore 
        # clr_t = tf.keras.models.load_model('models/clr_model.h5')
        # y_pred1 = clr_t.predict_classes(x_test_standard_lstm, batch_size=1)
        # print(classification_report(y_test, y_pred1))
        # tuned_t = tf.keras.models.load_model('models/tuned_ts_model.h5')
        # y_pred2 = tuned_t.predict_classes(x_test_standard_lstm, batch_size=1)
        # print(classification_report(y_test, y_pred2))
        # #
        return tuned_model_lstm

    def predict_with_lstm(self, model, x):
        """make prediction from lstm model

        Parameters
        ----------
        model: serializer
            the trained model
        x: DataFrame or np.array
            the input feature columns and target

        Return
        ----------
        pred : np.array
            the prediction, using boolean result
        pred_proba: np.array
            the probability result of each labels
        """
        pred = model.predict(x)
        pred_proba = model.predict_proba(x)

        return pred, pred_proba





































