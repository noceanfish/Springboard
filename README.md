Springboard
==============================

Springboard ML career track capestone project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── stk_predictor      <- Main project code here.
    │   └── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   └── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py  <- currently not used     
    │   └── models         <- code related to train the model, time-series and sentimental analysis
    │   │   └── predict_model.py  <- currently not used  
    │   │   └── train_sa_model.py  <- train sentimental analysis model  
    │   │   └── train_ts_model.py  <- train time-series analysis model 
    │   │   └── utils.py   <- help functions
    │   │   └── views.py   <- no use 
    │   └── predictor      <- the main function to make real-time prediction according to user's request
    │   │   └── forms.py   <- request forms validation, the home page related
    │   │   └── models.py  <- define price table in db
    │   │   └── views.py   <- action code for to response user request
    │   └── static         <- static file, including css, js, images
    │   └── templates      <- web site pages, html
    │   └── visualization  <- to generate necessary graph or other images
    │
    ├── app.py             <- Flask app initilization, the main entrance of app
    ├── commands.py        <- app.cli command, used for train model or init-db
    ├── compat.py          <- make the project compatible with different Python version
    ├── config.py          <- The app configuration
    ├── database.py        <- The db related operation
    ├── entensions.py      <- register flask extensions modules
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

# 1 Project Intro

This project implement NLP and time-series analysis on texts and stock trading data to get a specific stock's future movement.

## 1.1 Data Pipeline

The app scrape recent fresh data from Finviz and Yahoo Fiance. 

Then do the basic preprocessing and clean to the data set, including removing stopwords and special characters, expanding contractions, lemmatization, and tokenizer on texts data. 

In addition, the app generate log return, volatility, volumes with special time-window (5-day to 80-weeks).

## 1.2 Training the model

The app utilizes NLP embedding algorithms to get texts embeddings. We explor different embedding methos, word-count, tf-idf, gloves, bow, and elmo embedding. The last one give the hightest accuracy performance in predicting.

We also experiment different classification algorithms, such as LogisticRegression, SVM, Naive Bayes. Logistic Regression give better result.

To deal with time-eries model, we create a LSTM model and apply feature engineering results on the model.

## 1.3 Result 

Finally, we ensemble the two models to get the final results. the app tells people how will the stock move in the future based on nits recent news and historical price.

# 2 Data Storage

The app use sqllit3 to store price data; the news headlines are not persistence on local system; the original traing sets is save in csv files.

# 3 The Deployment

## 3.1 Using Docker

# 4 Screenshot
![Alt text](reports/home.png?raw=true "Optional Title")
![Alt text](reports/prediction-1.png?raw=true "Optional Title")
![Alt text](reports/prediction-2.png?raw=true "Optional Title")

The sentimental analysis apply ELMo
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
