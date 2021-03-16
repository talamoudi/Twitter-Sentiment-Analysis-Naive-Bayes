from utils import *





class Naive_Bayes_Model():
    
    def __init__(self):
        self.frequencies = dict()
        self.vocabulary = set()
        self.class_sums = [0,0]
        self.log_prior = 0
        self.log_likelihood = dict()
    


    def _generate_word_frequencies(self,tweets, labels):
        assert(len(tweets) == len(labels))
        for tweet, label in zip(tweets, labels):
            label = int(label)
            for word in tweet:
                if word not in self.vocabulary:
                    self.vocabulary.add(word)
                if (label, word) in self.frequencies:
                    self.frequencies[(label, word)] += 1
                else:
                    self.frequencies[(label, word)] = 1
                self.class_sums[label] += 1



    def predict(self, x, not_processed = True):
        if not_processed:
            x = process_tweets([x])[0]
        p = 0
        for word in x:
            if word in self.log_likelihood.keys():
                p += self.log_likelihood[word]

        return p + self.log_prior



    def train(self, X, Y, verbose = True):
        self._generate_word_frequencies(X, Y)
        Y = np.array(Y)
        self.log_prior = np.log(np.sum(Y) / (Y.shape[0] - np.sum(Y)))
        for word in self.vocabulary:
            count_in_positive_label = self.frequencies[(1, word)] if (1, word) in self.frequencies else 0
            count_in_negative_label = self.frequencies[(0, word)] if (0, word) in self.frequencies else 0
            prob_word_given_positive = (count_in_positive_label + 1.0) / (np.sum(Y == 1) + len(self.vocabulary))
            prob_word_given_negative = (count_in_negative_label + 1.0) / (np.sum(Y == 0) + len(self.vocabulary))
            likelihood = prob_word_given_positive / prob_word_given_negative
            self.log_likelihood[word] = np.log(likelihood)

        if verbose:
            print("The accuracy of the model on the training set is {}".format(
                                            self.test(X, Y, False)))

    
    def test(self, Xtest, Ytest, verbose = True):
        Yhat = list()
        for x in Xtest:
            label = 1 if self.predict(x, False) > 0 else 0
            Yhat.append(label)
        accuracy = np.mean( np.array(Yhat) == np.array(Ytest) )
        if verbose:
            print("The accuracy of the model on the test set is {}".format(accuracy))
        return accuracy
