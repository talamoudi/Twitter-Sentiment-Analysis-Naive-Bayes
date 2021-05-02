from naive_bayes_model import *





if __name__ == "__main__":
    model = load_model("./Results/twitter_sentiment_naive_bayes.model")
    line = input()
    while line:
        prediction = model.predict(line)
        label = "neutra"
        label = "positive" if prediction > 0 else label
        label = "negative" if prediction < 0 else label
        print(label)
        line = input()
