# Springboard
# 1. Data Sets:
     
## 1.1 ag_news_csv: 
* AG News data sets, 
* 31.3M
* From: https://registry.opendata.aws/fast-ai-nlp/
* Description: 496,835 categorized news articles from >2000 news sources
               from the 4 largest classes from AG’s corpus of news articles, using only the title and description fields.
               The number of training samples for each class is 30,000 and testing 1900.

## 1.2 nyt_comments: 
* New York Times Comments, 
* 1.55G
* From: https://www.kaggle.com/aashita/nyt-comments#ArticlesApril2017.csv
* Description: The data contains information about the comments made on the articles published in New York Times in Jan-May 2017 and Jan-April 2018.
               The month-wise data is given in two csv files - one each for the articles on which comments were made and for the comments themselves.
               The csv files for comments contain over 2 million comments in total with 34 features and those for articles contain 16 features about more than 9,000 articles.

## 1.3 one_week_newsfeeds:
* covers the 7 Day-period of August 24 through August 30 for the years 2017 and 2018. 
* 732M
* From: https://www.kaggle.com/therohk/global-news-week
* Description: Year 2017: 1,398,431 ; Year 2018: 1,912,872
               includes approximately 3.3 million articles, with 20,000 news sources and 20+ languages.
               four fields: publish_time, feed_code, source_url, headline_text

## 1.4 reuters21578: 
* the Reuters-21578 collection appeared on the Reuters newswire in 1987, 
* 28M
* From: http://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection
* Description: Number of Instances: 21578

## 1.5 sns_dataset
* “Multi-Source Social Feedback of Online News Feeds” 
* 129M
* From: http://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms#
* Description: a period of 8 months, between November 2015 and July 2016
               accounting for about 100,000 news items 
               4 different topics: economy, microsoft, obama and palestine.

## 1.6 sogou_news_csv
* news articles from the SogouCA and SogouCS news corpora
* 1.4G
* From: https://course.fast.ai/datasets
* Description: 2,909,551 news articles from the SogouCA and SogouCS news corpora, in 5 categories.
               training samples selected for each class is 90,000 and testing 12,000
               the Chinese characters have been converted to Pinyin.
               http://xzh.me/docs/charconvnet.pdf
               
## 1.7 training.1600000.processed.noemoticon
* Sentiment140 dataset with 1.6 million tweets
* 238M
* From: https://www.kaggle.com/kazanova/sentiment140
* Description: contains 1,600,000 tweets extracted using the twitter api
               annotated (0 = negative, 4 = positive) and they can be used to detect sentiment
               6 fields: target, ids, date, flag, user, text
               https://www.linkedin.com/pulse/social-machine-learning-h2o-twitter-python-marios-michailidis/
               
## 1.8 TimeSeries Data
* quandl-api get stock price 
* From: https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/usage/quickstart/python
* Description: using API to get stock price information

## 1.9 TimeSeries Data additional:
* Yahoo Finance
* From Yahoo API
* Description: https://rapidapi.com/blog/how-to-use-the-yahoo-finance-api/