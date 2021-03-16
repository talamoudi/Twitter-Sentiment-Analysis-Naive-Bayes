from naive_bayes_model import *





if __name__ == "__main__":
    Xtrain, Ytrain = fetch_data("./Data/training_data.csv", "./Data/training_labels.csv")
    Xtest, Ytest = fetch_data("./Data/testing_data.csv", "./Data/testing_labels.csv")
    model = Naive_Bayes_Model()
    model.train(Xtrain, Ytrain)
    model.test(Xtest, Ytest)
    save_model("./Results/twitter_sentiment_naive_bayes.model", model)
