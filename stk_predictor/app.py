# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
import logging
import os

from logging.handlers import RotatingFileHandler

from flask import Flask, render_template
from stk_predictor import commands, predictor, models, extensions

HERE = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.join(HERE, os.pardir)


def create_app():
    """Create application factory, as explained here:
       http://flask.pocoo.org/docs/patterns/appfactories/.

    :param
    """
    configure_logger()
    app = Flask(__name__.split('.')[0])

    # set config
    app_settings = os.getenv(
        "APP_SETTINGS", "stk_predictor.config.ProductionConfig"
    )
    app.config.from_object(app_settings)

    # register blueprint
    register_extensions(app)
    register_blueprints(app)
    register_errorhandlers(app)
    register_commands(app)
    return app


def register_blueprints(app):
    """Register Flask blueprints."""

    app.register_blueprint(predictor.views.blueprint)
    app.register_blueprint(models.views.blueprint)
    app.register_blueprint(commands.blueprint)
    return None


def register_extensions(app):
    """Register Flask extensions."""
    extensions.db.init_app(app)


def register_errorhandlers(app):
    """Register error templates."""

    def render_error(error):
        """Render error template."""

        # if a HTTPException, pull the 'code' attribute; default to 500
        error_code = getattr(error, "code", 500)
        return render_template(f"{error_code}.html"), error_code

    for errcode in [401, 404, 500]:
        app.errorhandler(errcode)(render_error)
    return None


def register_commands(app):
    """Register Click commands."""

    app.cli.add_command(commands.test)
    app.cli.add_command(commands.train_sentimental_model)
    app.cli.add_command(commands.train_time_series_model)
    app.cli.add_command(commands.init_db_command)


def configure_logger():
    """Configure loggers."""
    # handler = logging.StreamHandler(sys.stdout)
    log_dir = os.path.join(basedir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler(os.path.join(log_dir, 'flask.log'), maxBytes=1024*1024*10, backupCount=10)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    logging.getLogger().addHandler(handler)
    # if not app.logger.handlers:
    #     app.logger.addHandler(handler)
