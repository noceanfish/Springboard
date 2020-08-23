import feedparser
import pandas as pd
import os
import logging
from  datetime import datetime 

ITUNES = "http://ax.itunes.apple.com/WebObjects/MZStoreServices.woa/ws/RSS/topMovies/xml"
NETFLIX = ""
RSS_LIST = [ITUNES]

DF_COLUMNS = ['source', 'date', 'rank', 'title', 'link', 'summary']
logging.basicConfig(level=logging.DEBUG)


# create a function to return dataframe 
def top10_movies(rss, df):
    
    # parse the data 
    feed = feedparser.parse(rss)

    # check the bozo 
    if feed.bozo == 0:
        logging.info("%s has well-formed feed!" % feed.feed.title)
    else:
        logging.info("%s has flipped the bozo bit. Potential errors ahead!" % feed.feed.title)

    # get date
    feed_date = feed.feed.get('published', datetime.now().strftime('%Y-%m-%d'))
    
    i = 0
    while i < 10:
        feed_items = pd.Series([feed.feed.title, 
                                feed_date,
                                i+1,
                                feed.entries[i].title,
                                feed.entries[i].id,
                                feed.entries[i].summary], DF_COLUMNS)
        df = df.append(feed_items, ignore_index=True)
        i += 1

    return df


if __name__ == "__main__":
    # create a empty dataframe
    top10_df = pd.DataFrame(columns=DF_COLUMNS)
    
    # loop rss resources
    for item in RSS_LIST:
        top10_df = top10_movies(item, top10_df)

    if not os.path.isfile('top10.csv'):
        top10_df.to_csv('top10.csv', header=DF_COLUMNS, index=False)
    else:
        top10_df.to_csv('top10.csv', mode='a', header=False, index=False)
