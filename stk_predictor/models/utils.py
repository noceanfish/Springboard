# -*- coding: utf-8 -*-
# stk_predictor/model/utils.py
#

import unicodedata
from contractions import contractions_dict
import spacy
import nltk
from nltk.tokenize import ToktokTokenizer
import re
from bs4 import BeautifulSoup


class CorpusNormalization(object):
    """Normalize corpus"""

    def __init__(self):
        """Initialize some variables"""
        self.HASH_TAG_PATTERN = r'^@[a-zA-Z0-9]+'
        self.HYPERLINK_PATTERN = r'https?://[a-zA-Z0-9./]+'
        self.RE_PATTERN = r'|'.join([self.HASH_TAG_PATTERN, self.HYPERLINK_PATTERN])
        self.STOPWORDS = nltk.corpus.stopwords.words('english')
        self.STOPWORDS.remove('no')
        self.STOPWORDS.remove('not')
        self.nlp = spacy.load('en_core_web_md')
        self.tokenizer = ToktokTokenizer()

    def strip_html_tags(self, text):
        """clean html tag
        Parameters
        ----------
        text: str
              the input text

        Return
        ----------
        text: str
              the text that removes all html tag
        """
        soup = BeautifulSoup(text, 'lxml')
        stripped_text = soup.get_text()
        return stripped_text

    def remove_hash_tag_link(self, text):
        """Remove Hash Tag and hyper link in the text
        Parameters
        ----------
        text: str

        Return
        ----------
        text: str
              the text that removes all hash tag
        """
        text = re.sub(self.RE_PATTERN, '', text)
        return text

    def remove_accented_chars(self, text):
        """Removing accented character
        Parameters
        ----------
        text: str

        Return
        ----------
        text: str
              the text that removes all hash tag
        """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(self, text, contraction_mapping=contractions_dict):
        """expanding contractions
        Parameters
        ----------
        text: str
        contraction_mapping: dict
              mapping the contraction word to it original form

        Return
        ----------
        text: str
              the text that removes all hash tag

        """

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            if expanded_contraction:
                expanded_contraction = first_char + expanded_contraction[1:]
            else:
                expanded_contraction = match
            return expanded_contraction

        contraction_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                         flags=re.IGNORECASE | re.DOTALL)
        expanded_text = contraction_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", " ", expanded_text)
        return expanded_text

    def remove_special_characters(self, text):
        """remove special characters
        Parameters
        ----------
        text: str

        Return
        ----------
        text: str
              the text that just contains letter, digits, space

        """
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def lemmatize_text(self, text):
        """lemmztizing text
        Parameters
        ----------
        text: str

        Return
        ----------
        text: str
              the word in the text was substitute by its orginal form

        """
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def remove_stopwords(self, text, is_lower_case=False):
        """remove stopwords
        Parameters
        ----------
        text: str
        is_lower_case: boolean

        Return
        ----------
        text: str
              remove all stop words in STOPWORDSLIST

        """
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]

        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.STOPWORDS]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.STOPWORDS]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def transform(self, corpus, html_stripping=True, hash_link=True,
                  contraction_expansion=True, accented_char_removal=True,
                  text_lower_case=True, text_lemmatization=True,
                  special_char_removal=True, stopword_removal=True):
        """
        Parameters
        ----------
        corpus: list, each element is an str
        transform switch: boolean
            True: do the transform
            False: ignore this step

        Return
        ----------
        normalized_corpus: list
              the normalized corpus list

        """
        normalized_corpus = []

        for doc in corpus:
            if html_stripping:
                doc = self.strip_html_tags(doc)
            if hash_link:
                doc = self.remove_hash_tag_link(doc)
            if accented_char_removal:
                doc = self.remove_accented_chars(doc)
            if contraction_expansion:
                doc = self.expand_contractions(doc)
            if text_lower_case:
                doc = doc.lower()

            doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)

            if text_lemmatization:
                doc = self.lemmatize_text(doc)
            if special_char_removal:
                doc = self.remove_special_characters(doc)

            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)

            if stopword_removal:
                doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)

            normalized_corpus.append(doc)

        return normalized_corpus
