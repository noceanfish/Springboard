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

    # # logarithmic return for 1-day
    # log_ret_1d = Column(db.Float, nullable=False)

    # # logarithmic return on 23 different periods
    # log_ret_1w = Column(db.Float, nullable=False)
    # log_ret_2w = Column(db.Float, nullable=False)
    # log_ret_3w = Column(db.Float, nullable=False)
    # log_ret_4w = Column(db.Float, nullable=False)
    # log_ret_8w = Column(db.Float, nullable=False)
    # log_ret_12w = Column(db.Float, nullable=False)
    # log_ret_16w = Column(db.Float, nullable=False)
    # log_ret_20w = Column(db.Float, nullable=False)
    # log_ret_24w = Column(db.Float, nullable=False)
    # log_ret_28w = Column(db.Float, nullable=False)
    # log_ret_32w = Column(db.Float, nullable=False)
    # log_ret_36w = Column(db.Float, nullable=False)
    # log_ret_40w = Column(db.Float, nullable=False)
    # log_ret_44w = Column(db.Float, nullable=False)
    # log_ret_48w = Column(db.Float, nullable=False)
    # log_ret_52w = Column(db.Float, nullable=False)
    # log_ret_56w = Column(db.Float, nullable=False)
    # log_ret_60w = Column(db.Float, nullable=False)
    # log_ret_64w = Column(db.Float, nullable=False)
    # log_ret_68w = Column(db.Float, nullable=False)
    # log_ret_72w = Column(db.Float, nullable=False)
    # log_ret_76w = Column(db.Float, nullable=False)
    # log_ret_80w = Column(db.Float, nullable=False)

    # # volatility, 23 col
    # vol_1w = Column(db.Float, nullable=False)
    # vol_2w = Column(db.Float, nullable=False)
    # vol_3w = Column(db.Float, nullable=False)
    # vol_4w = Column(db.Float, nullable=False)
    # vol_8w = Column(db.Float, nullable=False)
    # vol_12w = Column(db.Float, nullable=False)
    # vol_16w = Column(db.Float, nullable=False)
    # vol_20w = Column(db.Float, nullable=False)
    # vol_24w = Column(db.Float, nullable=False)
    # vol_28w = Column(db.Float, nullable=False)
    # vol_32w = Column(db.Float, nullable=False)
    # vol_36w = Column(db.Float, nullable=False)
    # vol_40w = Column(db.Float, nullable=False)
    # vol_44w = Column(db.Float, nullable=False)
    # vol_48w = Column(db.Float, nullable=False)
    # vol_52w = Column(db.Float, nullable=False)
    # vol_56w = Column(db.Float, nullable=False)
    # vol_60w = Column(db.Float, nullable=False)
    # vol_64w = Column(db.Float, nullable=False)
    # vol_68w = Column(db.Float, nullable=False)
    # vol_72w = Column(db.Float, nullable=False)
    # vol_76w = Column(db.Float, nullable=False)
    # vol_80w = Column(db.Float, nullable=False)

    # # volumes mean
    # volume_1w = Column(db.Float, nullable=False)
    # volume_2w = Column(db.Float, nullable=False)
    # volume_3w = Column(db.Float, nullable=False)
    # volume_4w = Column(db.Float, nullable=False)
    # volume_8w = Column(db.Float, nullable=False)
    # volume_12w = Column(db.Float, nullable=False)
    # volume_16w = Column(db.Float, nullable=False)
    # volume_20w = Column(db.Float, nullable=False)
    # volume_24w = Column(db.Float, nullable=False)
    # volume_28w = Column(db.Float, nullable=False)
    # volume_32w = Column(db.Float, nullable=False)
    # volume_36w = Column(db.Float, nullable=False)
    # volume_40w = Column(db.Float, nullable=False)
    # volume_44w = Column(db.Float, nullable=False)
    # volume_48w = Column(db.Float, nullable=False)
    # volume_52w = Column(db.Float, nullable=False)
    # volume_56w = Column(db.Float, nullable=False)
    # volume_60w = Column(db.Float, nullable=False)
    # volume_64w = Column(db.Float, nullable=False)
    # volume_68w = Column(db.Float, nullable=False)
    # volume_72w = Column(db.Float, nullable=False)
    # volume_76w = Column(db.Float, nullable=False)
    # volume_80w = Column(db.Float, nullable=False)


    # def __init__(self, trading_date, intraday_close, intraday_volumes, log_ret_1d,
    #         log_ret_1w, log_ret_2w, log_ret_3w, log_ret_4w, log_ret_8w, log_ret_12w, 
    #         log_ret_16w, log_ret_20w, log_ret_24w, log_ret_28w, log_ret_32w, log_ret_36w,
    #         log_ret_40w, log_ret_44w, log_ret_48w, log_ret_52w, log_ret_56w, log_ret_60w,
    #         log_ret_64w, log_ret_68w, log_ret_72w, log_ret_76w, log_ret_80w,
    #         vol_1w, vol_2w, vol_3w, vol_4w, vol_8w, vol_12w, vol_16w, vol_20w,
    #         vol_24w, vol_28w, vol_32w, vol_36w, vol_40w, vol_44w, vol_48w, vol_52w,
    #         vol_56w, vol_60w, vol_64w, vol_68w, vol_72w, vol_76w, vol_80w,
    #         volume_1w, volume_2w, volume_3w, volume_4w, volume_8w, volume_12w, 
    #         volume_16w, volume_20w, volume_24w, volume_28w, volume_32w, volume_36w, 
    #         volume_40w, volume_44w, volume_48w, volume_52w, volume_56w, volume_60w, 
    #         volume_64w, volume_68w, volume_72w, volume_76w, volume_80w, **kwargs
    # 	):
    def __init__(self, trading_date, intraday_close, intraday_volumes):
        self.trading_date = trading_date
        self.intraday_close = intraday_close
        self.intraday_volumes = intraday_volumes
        
        # self.log_ret_1d = log_ret_1d

        # self.log_ret_1w = log_ret_1w
        # self.log_ret_2w = log_ret_2w
        # self.log_ret_3w = log_ret_3w
        # self.log_ret_4w = log_ret_4w
        # self.log_ret_8w = log_ret_8w
        # self.log_ret_12w = log_ret_12w
        # self.log_ret_16w = log_ret_16w
        # self.log_ret_20w = log_ret_20w
        # self.log_ret_24w = log_ret_24w
        # self.log_ret_28w = log_ret_28w
        # self.log_ret_32w = log_ret_32w
        # self.log_ret_36w = log_ret_36w
        # self.log_ret_40w = log_ret_40w
        # self.log_ret_44w = log_ret_44w
        # self.log_ret_48w = log_ret_48w
        # self.log_ret_52w = log_ret_52w
        # self.log_ret_56w = log_ret_56w
        # self.log_ret_60w = log_ret_60w
        # self.log_ret_64w = log_ret_64w
        # self.log_ret_68w = log_ret_68w
        # self.log_ret_72w = log_ret_72w
        # self.log_ret_76w = log_ret_76w
        # self.log_ret_80w = log_ret_80w

        # # volatility, 23 col
        # self.vol_1w = vol_1w
        # self.vol_2w = vol_2w
        # self.vol_3w = vol_3w
        # self.vol_4w = vol_4w
        # self.vol_8w = vol_8w
        # self.vol_12w = vol_12w
        # self.vol_16w = vol_16w
        # self.vol_20w = vol_20w
        # self.vol_24w = vol_24w
        # self.vol_28w = vol_28w
        # self.vol_32w = vol_32w
        # self.vol_36w = vol_36w
        # self.vol_40w = vol_40w
        # self.vol_44w = vol_44w
        # self.vol_48w = vol_48w
        # self.vol_52w = vol_52w
        # self.vol_56w = vol_56w
        # self.vol_60w = vol_60w
        # self.vol_64w = vol_64w
        # self.vol_68w = vol_68w
        # self.vol_72w = vol_72w
        # self.vol_76w = vol_76w
        # self.vol_80w = vol_80w
    
        # # volumes mean
        # self.volume_1w = volume_1w
        # self.volume_2w = volume_2w
        # self.volume_3w = volume_3w
        # self.volume_4w = volume_4w
        # self.volume_8w = volume_8w
        # self.volume_12w = volume_12w
        # self.volume_16w = volume_16w
        # self.volume_20w = volume_20w
        # self.volume_24w = volume_24w
        # self.volume_28w = volume_28w
        # self.volume_32w = volume_32w
        # self.volume_36w = volume_36w
        # self.volume_40w = volume_40w
        # self.volume_44w = volume_44w
        # self.volume_48w = volume_48w
        # self.volume_52w = volume_52w
        # self.volume_56w = volume_56w
        # self.volume_60w = volume_60w
        # self.volume_64w = volume_64w
        # self.volume_68w = volume_68w
        # self.volume_72w = volume_72w
        # self.volume_76w = volume_76w
        # self.volume_80w = volume_80w


    def __repr__(self):
        """Represent instance as a unique string."""
        return f"<Apple({self.id!r})>"















