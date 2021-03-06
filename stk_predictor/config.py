# -*- coding: utf-8 -*-
"""Application configuration.

Most configuration is set via environment variables.

For local development, use a .env file to set
environment variables.
"""


import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       os.pardir))

class BaseConfig(object):
    """Base configuration"""

    db_dir = os.path.join(basedir, 'db')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    APP_NAME = os.getenv("APP_NAME", "stk_predictor")
    DEBUG_TB_ENABLED = False
    SECRET_KEY = os.getenv("SECRET_KEY", "secret_key_here")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False


class DevelopmentConfig(BaseConfig):
    """Development configuration."""

    DEBUG_TB_ENABLED = True
    DEBUG_TB_INTERCEPT_REDIRECTS = False
    ENV = "development"
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        "sqlite:///{0}".format(os.path.join(BaseConfig.db_dir, "dev.db"))
    )


class TestingConfig(BaseConfig):
    """Testing configuration."""

    PRESERVE_CONTEXT_ON_EXCEPTION = False
    TESTING = True


class ProductionConfig(BaseConfig):
    """Production configuration."""

    WTF_CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        "sqlite:///{0}".format(os.path.join(BaseConfig.db_dir, "prod.db"))
    )
