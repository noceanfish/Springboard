# -*- coding: utf-8 -*-
"""predictor views."""

import os
import pickle
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import json

from sqlalchemy import func

from flask import (Blueprint,
                   current_app,
                   render_template,
                   flash,
                   redirect,
                   request,
                   url_for)
from stk_predictor.predictor.forms import MakePredictionForm
from stk_predictor.predictor.models import Apple
from stk_predictor.utils import flash_errors
from stk_predictor.data import make_dataset
from stk_predictor.models import train_sa_model, train_ts_model
from stk_predictor.extensions import db
from stk_predictor.visualization import visualize

blueprint = Blueprint("predictor", __name__)
HERE = os.path.abspath(os.path.dirname(__file__))


@blueprint.route("/", methods=["GET", "POST"])
def home():
    """Home page"""
    form = MakePredictionForm(request.form)
    current_app.logger.info("User request Home page.")

    return render_template("/home.html", form=form)


@blueprint.route("/about", methods=["GET", "POST"])
def about():
    """Home page"""
    # form = MakePredictionForm(request.form)
    current_app.logger.info("User request About page.")

    return render_template("/about.html")


@blueprint.route("/predict", methods=["GET", "POST"])
def predict():
    """Make a prediction according user's ticker"""
    current_app.logger.info("make prediction request.")
    project_parent = '/'.join(current_app.root_path.split('/')[:-1])

    if request.method == "POST":
        form = MakePredictionForm(request.form)
        if form.validate_on_submit():
            current_app.logger.info("successful validate the form. continue to make prediction.")

            # load the sentimental analysis model
            try:
                # get ticket_name from input form
                ticker = form.ticker_name.data.upper()

                # retrieve Sentimental analysis model
                sa_model_path = os.path.join(project_parent, "models/sa_classifier.pkl")
                if os.path.exists(sa_model_path):
                    with open(sa_model_path, 'rb') as f:
                        sa_classifier = pickle.load(f)
                else:
                    flash("Sorry, Sentimental analysis model path does not exist.", "Failed")
                    flash_errors(form)
                    return render_template("/result.html", form=form)

                # retrieve time-series model
                ts_model_path = os.path.join(project_parent, "models/tuned_ts_model.h5")
                if os.path.exists(ts_model_path):
                    ts_classifier = tf.keras.models.load_model(ts_model_path)
                else:
                    flash("Sorry, time series model path does not exist.", "Failed")
                    flash_errors(form)
                    return render_template("/result.html", form=form)

                # get raw data from public resource
                recent_trading = db.session.query(func.max(Apple.trading_date)).scalar()
                start = recent_trading
                end = datetime.datetime.today()
                price_df = make_dataset.get_ticker_from_yahoo(ticker, start, end)

                trading_features_df = make_dataset.prepare_trading_dataset(price_df.iloc[-401:])
                ts_pred = ts_classifier.predict_classes(trading_features_df, batch_size=1)[0]
                ts_pred_prob = ts_classifier.predict(trading_features_df, batch_size=1)[0]
                # trading_features_df['pred'] = ts_pred
                # trading_features_df['pred_prob'] = ts_pred_prob

                # predict news sentimental
                # first get news data from Finviz
                # then, make prediction using pre-trained model
                news_df = make_dataset.get_news_from_finviz(ticker)

                sa_obj = train_sa_model.SentimentalAnalysisModel()
                cleaned_news = sa_obj.dataset_preprocessing(news_df.texts.values)
                feature_embeddings = sa_obj.generate_elmo_embeddings(cleaned_news)

                # get prediction result: predicting classes, probability
                sa_pred = sa_classifier.predict(feature_embeddings)
                sa_pred_prob = sa_classifier.predict_proba(feature_embeddings)
                # add result to news_df
                news_df['pred'] = sa_pred
                news_df['pred_prob'] = [i[np.argmax(i)] for i in sa_pred_prob]

                # calculate the pos/neg/neu statistic info
                total_news_pos = len(news_df[news_df['pred'] == 1])
                total_news_neg = len(news_df[news_df['pred'] == -1])
                total_news_neu = len(news_df[news_df['pred'] == 0])
                num = len(news_df[~(news_df['pred'] == 0)])
                final_res2 = np.sum(np.abs(news_df['pred'].astype(float).values) * news_df['pred_prob'].values) \
                             / np.abs(total_news_neg + total_news_pos)
                final_res2 = final_res2 * (0.7 if (total_news_pos - total_news_neg) >= 0 else -0.7)

                # change label classes to LABEL
                news_df.loc[news_df['pred'] == 1, 'pred'] = "POS"
                news_df.loc[news_df['pred'] == -1, 'pred'] = "NEG"
                news_df.loc[news_df['pred'] == 0, 'pred'] = "NEU"

                # get the result and visualization
                plot = visualize.plot_history_price(ticker.upper(), price_df)

                # ensemble the result
                # simply, we just add weighted positive and negative sentimental to get a final sentimental result
                # since the sentimental analysis can reach 91% correction, we highlight 0.7 weight on sentimental
                # result, and gives 0.3 weight to price result.
                final_res1 = ts_pred_prob[0] * 0.3 * (1 if ts_pred[0] > 0 else -1)

                fin = final_res1 + final_res2

                ## test for rende template
                # news_df.to_csv('news_df.csv', index=False)
                # plot_name = 'AAPL20-09-25_08-45-33.png'
                # news_df = pd.read_csv('news_df.csv')
                # total_news_pos = len(news_df[news_df['pred'] == 1])
                # total_news_neg = len(news_df[news_df['pred'] == -1])
                # total_news_neu = len(news_df[news_df['pred'] == 0])
                # news_df.loc[news_df['pred'] == 1, 'pred'] = "POS"
                # news_df.loc[news_df['pred'] == -1, 'pred'] = "NEG"
                # news_df.loc[news_df['pred'] == 0, 'pred'] = "NEU"
                # fin = -0.4033
                # ts_pred = np.array([1])
                # ts_pred_prob = np.array([0.77])

                res_data = {
                    'plot_img': plot,
                    'news_df': news_df.to_dict(orient='records'),
                    'fin': fin,
                    'pri_res': ts_pred,
                    'pri_pred_prob': ts_pred_prob,
                    'total_news_pos': total_news_pos,
                    'total_news_neg': total_news_neg,
                    'total_news_neu': total_news_neu
                    }

                return render_template("/result.html", form=form, res_data=res_data)

            except Exception as ex:
                flash("Sorry, Something go wrong! {}".format(ex), "Failed")
                current_app.logger.error("Sorry, Got exception", ex)
        else:
            flash_errors(form)
    return render_template("/result.html", form=form)



