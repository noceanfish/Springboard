# -*- coding: utf-8 -*-
# stk_predictor/model/views.py
#

from flask import Blueprint, current_app, render_template, flash, redirect, request, url_for
from stk_predictor.predictor.forms import MakePredictionForm


blueprint = Blueprint("train_newmodel", __name__, url_prefix="/trainNewmodel")


@blueprint.route("/ts_model", methods=["GET", "POST"])
def ts_model():
    """Home page"""
    # form = MakePredictionForm(request.form)
    current_app.logger.info("Train new time-series model according to user request.")

    return render_template("/train.html")