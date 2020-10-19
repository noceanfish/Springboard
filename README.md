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

This project implements NLP and time-series analysis on texts and stock trading data to get a specific company's stock future movement.

## 1.1 Data Pipeline

The app scrapes recent fresh data from Finviz and Yahoo Fiance. 

Then do the basic preprocessing and cleaning to the data set, including removing stop-words and special characters, expanding contractions, lemmatization, and tokenizer on texts data. 

In addition, in order to apply classification on time-series data, the app generates log return, volatility, volumes with special time-window (5-day to 80-weeks) as input features. And I design a LSTM model to achieve classification result on time series data. 

## 1.2 Training the model

The app utilizes NLP embedding algorithms to get texts embeddings. I explore different embedding methods, including word-count, tf-idf, gloves, bow, and elmo embedding. The last one gives the hightest accuracy performance on predicting the news sentimental.

I also experiment different classification algorithms, such as LogisticRegression, SVM, Naive Bayes. Logistic Regression give better result.

To deal with time-eries model, I create a LSTM model and apply feature engineering results on the model.

## 1.3 Result 

Finally, I ensemble the two model's judgement to get the final results. The app tells people how will the stock move in the future based on its recent news and historical price.

# 2 Data Storage

* The app use sqllit3 to store price data; 
* the news headlines are not persistence on local system; 
* the original traing sets is saved in csv files.

# 3 The Deployment

## 3.1 Using Docker to initialize the db
    * this will save the logs and db in disk-volumes as mounted in command
```
docker run -it \
  -v '/local/storage/db:/app/db' \
  -v '/local/storage/logs:/app/logs' \
  springboard-csp init-db
```

## 3.2 run the app
```
docker run -it -p 5000:5000 
   -v '/local/storage/db:/app/db' \
   -v '/local/storage/logs:/app/logs' \
   springboard-csp
```

# 4 Screenshot
![Alt text](reports/home.png?raw=true "Optional Title")
![Alt text](reports/prediction-1.png?raw=true "Optional Title")
![Alt text](reports/prediction-2.png?raw=true "Optional Title")

# 5 How to use

## 5.1 run the app
when you deploy the app with production server, such as uwsgi, 
In experiment environment, we can run the app by flask interal server:
```
export FLASK_ENV=production
export FLASK_APP=stk_predictor.app
flask run
```

## 5.2 how to train the classifier
If you want to predict a new stock, you have to train a new model by using the cli command:
```
flask train_ts_model AAPL // it will load AAPL history data and train a new model

flask train_sa_model // this will bring up training sentimental model
``` 
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
