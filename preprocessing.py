import os
import random

from utils import *





if __name__ == "__main__":

    user = os.path.expanduser('~')
    if not os.path.isdir(user + '/nltk_data/corpora/twitter_samples'):
        nltk.download('twitter_samples')
    
    if not os.path.isdir(user + '/nltk_data/corpora/stopwords'):
        nltk.download('stopwords')

    if not os.path.isdir(user + '/nltk_data/tokenizers'):
        nltk.download('tokenizers')

    pos, neg = get_tweets()

    pos_tweets = process_tweets(pos)
    random.shuffle(pos_tweets)
    index = int(len(pos_tweets) * 0.1) + 1
    testing_data = pos_tweets[:index]
    training_data = pos_tweets[index:]
    training_labels = [1]*len(training_data)
    testing_labels = [1]*len(testing_data)

    neg_tweets = process_tweets(neg)
    random.shuffle(neg_tweets)
    index = int(len(neg_tweets) * 0.1) + 1
    testing_data.extend( neg_tweets[:index] )
    training_data.extend( neg_tweets[index:] )
    training_labels.extend([0]*len(neg_tweets[index:]))
    testing_labels.extend([0]*len(neg_tweets[:index]))

    assert(len(training_labels) == len(training_data))
    assert(len(testing_labels) == len(testing_data))

    write_data('./Data/training_data.csv', training_data)
    write_data('./Data/testing_data.csv', testing_data)
    write_data('./Data/training_labels.csv', training_labels)
    write_data('./Data/testing_labels.csv', testing_labels)