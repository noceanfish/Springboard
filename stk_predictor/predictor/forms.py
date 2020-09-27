# -*- coding: utf-8 -*-
# stk_predictor/predictor/forms.py
#

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length


class MakePredictionForm(FlaskForm):
    """Make prediction form."""

    ticker_name = StringField(
        "Ticker Name", 
        default='AAPL',
        validators=[DataRequired(), Length(min=1, max=15)]
    )

    def __init__(self, *args, **kwargs):
        """Create instance"""
        super(MakePredictionForm, self).__init__(*args, **kwargs)

    def validate(self):
        """Validate the form."""
        initial_validation = super(MakePredictionForm, self).validate()

        if not initial_validation:
            self.ticker_name.errors.append("Unknown tickername.")
            return False
        if self.ticker_name.data.upper() != "AAPL":
            self.ticker_name.errors.append(
                "Current real-time prediction only support 'APPL'.")
            return False

        return True
