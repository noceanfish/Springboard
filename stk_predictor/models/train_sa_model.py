# -*- coding: utf-8 -*-
# stk_predictor/model/train_sa_model.py
#

import re
import pickle
import numpy as np

# ML algorithm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# evaluation metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

import spacy
import tensorflow as tf
import tensorflow_hub as hub

from flask import current_app
from .utils import CorpusNormalization


class SentimentalAnalysisModel(object):
    """Generate sentimental analysis model
    using ELMo and lr
    """

    def __init__(self, 
                 feature_size=1024, 
                 test_size=0.3, 
                 elmo_output_name='elmo'):
        """Preprocess the Dataset
        Parameters
        ----------
        feature_size: DataFrame
            the elmo output feature size
        test_size: int
            the train test split ratio
        elmo_output_name: str
            the elmo output layer name

        Return
        ----------
        df: DataFrame
            the cleaned df
        """

        # clean the texts, expand money-unit: mn -> million
        # lower the words, replace '\t' '\n' '%'
        self.feature_size = feature_size
        self.test_size = test_size

        # initialize tf variables
        url = "https://tfhub.dev/google/elmo/3"
        elmo_model = hub.Module(url, trainable=False)
        self.sess = tf.compat.v1.Session()

        self.tokens_ph = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
        self.tokens_length_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.int32, name='tokens_length')
        self.elmo_output = elmo_model(inputs={"tokens": self.tokens_ph,
                                              "sequence_len": self.tokens_length_ph},
                                      signature="tokens", 
                                      as_dict=True)[elmo_output_name]
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def dataset_preprocessing(self, texts):
        """Preprocess the Dataset
        Parameters
        ----------
        texts: DataFrame
            the input dataset features columns and label column
        targets: DataFrame
            the target labels

        Return
        ----------
        df: DataFrame
            the cleaned df
        """

        # clean the texts, expand money-unit: mn -> million
        # lower the words, replace '\t' '\n' '%'
        normalizer = CorpusNormalization()
        current_app.logger.info("Enter dataset_preprocessing step...")

        pattern = re.compile('(\smn\s)')

        # load spacy model
        # nlp = spacy.load('en_core_web_md')
        cleaned_text = []

        for line in texts:
            line = line.lower()
            line = line.replace('\n', ' ').replace('\t', ' ')
            line = line.replace('\xa0', ' ').replace('%', 'percent')
            line = re.sub(pattern, ' millions ', line)
            line = ' '.join(line.split())

            line = normalizer.expand_contractions(line)
            line = normalizer.lemmatize_text(line)
            line = normalizer.remove_special_characters(line)
            line = normalizer.remove_stopwords(line, is_lower_case=True)

            tokens = normalizer.tokenizer.tokenize(line)
            tokens = [token.strip() for token in tokens]

            cleaned_text.append(tokens)
        current_app.logger.info(
            "Job >> dataset_preprocessing finished. Totally preprocessed {}"
            .format(len(cleaned_text))
        )

        return cleaned_text

    def dataset_split(self, dataset, targets):
        """split the dataset for training

        Parameters
        ----------
        dataset: DataFrame or array
            cleaned dataset, contains news texts
        target: DataFrame
            the label column

        Return
        ----------
        filled_batch : list
            the embedded feature matrix
        tokens_length: int
            the max length of tokens among batch lines
        """
        x_train, x_test, y_train, y_test = train_test_split(
            dataset, targets, test_size=self.test_size
        )

        current_app.logger.info(
            "Dataset infor: X_train shape is: {}; X_test shape is: {}"
            .format(len(x_train), len(x_test))
        )

        return x_train, x_test, y_train, y_test

    def fill_batch(self, batch):
        """Filled each batch line with nan, to meet the max length

        Parameters
        ----------
        batch: DataFrame or array
            a list of tokenized texts

        Return
        ----------
        filled_batch : list
            the embedded feature matrix
        tokens_length: int
            the max length of tokens among batch lines
        """
        if not batch:
            empty_vec = np.zeros(self.feature_size, dtype=np.float32)
            return empty_vec

        filled_batch = []
        for line in batch:
            line = line if line else ['']
            filled_batch.append(line)

        tokens_length = [len(batch_line) for batch_line in filled_batch]
        tokens_length_max = max(tokens_length)
        filled_batch = [batch_line + [''] * (tokens_length_max - len(batch_line)) for batch_line in filled_batch]

        return filled_batch, tokens_length

    def generate_elmo_embeddings(self, texts):
        """Preprocess the Dataset embedding by ELMo pre_trained model
        from google's tensor-hub

        Parameters
        ----------
        texts: DataFrame or array
            the input feature columns

        Return
        ----------
        elmo_embeddings : list
            the embedded feature matrix
        """
        current_app.logger.info("Enter elmo embeddings step...")

        total = len(texts)
        batch_size = min(total, 100)
        # counter = 0

        elmo_embeddings = []
        for i in range(0, total, batch_size):
            # counter += 1
            batch = texts[i:min(total, i+batch_size)]
            input_batch, input_length = self.fill_batch(batch)
            x = self.sess.run(tf.reduce_mean(self.elmo_output, 1),
                              feed_dict={self.tokens_ph: input_batch,
                                         self.tokens_length_ph: input_length})
            elmo_embeddings.extend(x)
            current_app.logger.info("Job >> generate elmo embeddings...{}/{}".format(min(i+batch_size, total), total))

        current_app.logger.info("Finish elmo embeddings step. The total samples are {}".format(len(elmo_embeddings)))
        return elmo_embeddings

    def run_logistic_regression(self, x_train, x_test, y_train, y_test):
        """By utilizing Dataset embedding, train the model by using LinearRergression
        from sklearn

        Parameters
        ----------
        x_train, x_test, y_train, y_test: DataFrame or np.array
            the input feature columns and target

        Return
        ----------
        classifier : sklearn model
            the trained classifier model
        """
        current_app.logger.info("Enter training step, using LogisticRegression...")

        param_grid = {
            'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
            'C': [2**x for x in range(3, 7)],
            'penalty': ['l2']
        }

        lr = LogisticRegression()
        gs = GridSearchCV(estimator=lr, 
                          param_grid=param_grid,
                          scoring='accuracy',
                          n_jobs=1,
                          cv=10)
        gs.fit(x_train, y_train)
        accuracy = accuracy_score(y_true=y_test, y_pred=gs.predict(x_test))
        current_app.logger.info('The best estimator accuracy is {:.1f}%'.format(accuracy * 100))
        current_app.logger.info(gs.best_estimator_)
        current_app.logger.info(gs.best_params_)

        with open('models/sa_model.pkl', 'wb') as f:
            pickle.dump(gs.best_estimator_, f)

        return gs.best_estimator_

    def predict_with_lr(self, model, x):
        """By utilizing Dataset embedding, train the model by using LinearRergression
        from sklearn

        Parameters
        ----------
        model: serializer
            the trained model
        x: DataFrame or np.array
            the input feature columns and target

        Return
        ----------
        pred : np.array
            the prediction, using boolean result
        pred_proba: np.array
            the probability result of each labels
        """
        pred = model.predict(x)
        pred_proba = model.predict_proba(x)

        return pred, pred_proba