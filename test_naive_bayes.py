from naive_bayes_model import *





if __name__ == "__main__":
    model = load_model("./Results/twitter_sentiment_naive_bayes.model")
    Xtest, Ytest = fetch_data("./Data/testing_data.csv", "./Data/testing_labels.csv")
    model.test(Xtest, Ytest)