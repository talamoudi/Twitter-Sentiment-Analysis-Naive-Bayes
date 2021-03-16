import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump

import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import re
import string

from itertools import chain





def get_tweets(pos_path='positive_tweets.json',neg_path='negative_tweets.json'):
    positive_tweets = twitter_samples.strings(pos_path)
    negative_tweets = twitter_samples.strings(neg_path)
    return positive_tweets, negative_tweets





def process_tweets(tweets):
    processed_tweets = list()
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    st_words = stopwords.words('english')
    stemmer = PorterStemmer() 
    
    for tweet in tweets:
        new_tweet = tweet.replace('\n','')
        new_tweet = re.sub(r'^RT[\s]+', '', new_tweet)
        new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)
        new_tweet = re.sub(r'#', '', new_tweet)
        tweet_tokens = tokenizer.tokenize(new_tweet)
        new_tokens = list()
        for word in tweet_tokens:
            if word not in st_words and word not in string.punctuation:
                new_tokens.append( stemmer.stem(word) )
        if len(new_tokens) > 0:
            processed_tweets.append(new_tokens)

    return processed_tweets





def read_data(filename):
    Data = list()
    infile = open(filename, "r")
    for line in infile:
        line = line[:-1]
        tokens = line.split(",")
        if len(tokens) == 0:
            continue
        Data.append(tokens)
    infile.close()
    return Data





def write_data(filename, data):
    file = open(filename,"w")
    for item in data:
        if type(item) == list:
            for i in range(len(item)):
                if item[i] == '\n':
                    continue
                file.write(item[i])
                if i == len(item) - 1:
                    file.write("\n")
                else:
                    file.write(",")
        else:
            file.write(str(item) + "\n")
    file.close()





def save_model(filename, model):
    file = open(filename, 'wb')
    dump(model, file)
    file.close()





def load_model(filename):
    file = open(filename, 'rb')
    model = load(file)
    file.close()
    return model





def fetch_data(training_file, testing_file):
    X = read_data(training_file)
    Y = read_data(testing_file)
    Y = list(chain.from_iterable(Y))
    Y = [int(y) for y in Y]
    return X, Y