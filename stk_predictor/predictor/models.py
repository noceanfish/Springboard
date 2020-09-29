# -*- coding: utf-8 -*-
"""Ticker Price models."""
from stk_predictor.database import (
    Column,
    PkModel,
    db,
    reference_col,
    relationship,
)


class Apple(PkModel):
    """Company-AAPL history price table"""

    __tablename__ = "aapl"
    trading_date = Column(db.DateTime, primary_key=True, unique=True, nullable=False)
    # trading_date = Column(db.DateTime, nullable=False)
    intraday_close = Column(db.Float, nullable=False)
    intraday_volumes = Column(db.Float, nullable=False)

    def __init__(self, ids, trading_date, intraday_close, intraday_volumes):
        self.id = ids
        self.trading_date = trading_date
        self.intraday_close = intraday_close
        self.intraday_volumes = intraday_volumes

    def __repr__(self):
        """Represent instance as a unique string."""
        return f"<Apple({self.id!r})>"















