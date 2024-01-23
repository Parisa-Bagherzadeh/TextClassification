import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser
import tensorflow as tf




class EmojiTextClassifier:
    def __init__(self, dim) -> None:
        self.dim = dim


    def load_dataset(self, dataset_path):
        df = pd.read_csv(dataset_path)
        X = np.array(df["sentence"].to_numpy())
        Y = np.array(df["label"].to_numpy(), dtype = int)
        return X, Y
    
    def load_feature_vector(self,vector_file):
        self.word_vectors = {}

        for line in vector_file:
            line = line.strip().split(" ")
            word = line[0]
            vector = np.array(line[1:], dtype = np.float64)
            self.word_vectors[word] = vector

    def sentence_to_feature_vectors_avg(self, sentence):
        
        try:
            sentence = sentence.lower()
            words = sentence.strip().split(" ")
            sum_vectors = np.zeros((self.dim, ))
            for word in words:
                sum_vectors += self.word_vectors[word]

            self.avg_vector = sum_vectors / len(words)
        
            return self.avg_vector
        
        except:
            print(sentence) 



    def load_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, 
                                  input_shape = (self.dim,),
                                  activation = "softmax")
        ])

        self.model.compile(
            tf.keras.optimizers.Adam(),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )


    def train(self, X_train, Y_train):
        output = self.model.fit(X_train, Y_train, epochs = 400)
        return output

    def test(self, X_test, Y_test):
        result = self.model.evaluate(X_test, Y_test)    
        return result

    def label_to_emoji(self, label):
        emojies = ["❤️","⚽️","😄","😔","🍽"]
        return emojies[label]
    

    def inference(self, test_sentence):
        sentence = self.sentence_to_feature_vectors_avg(test_sentence)
        sentence  = np.array([sentence])
        result = self.model.predict(sentence)
        y_pred = np.argmax(result)
        label = self.label_to_emoji(y_pred)

        return label

        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dimension", type = int, help="dimension of feature vector", default = 300)
    parser.add_argument("--sentence",type = str, 
                        help="example sentence to determine its class", 
                        default="I like pizza")
    args = parser.parse_args()

    train_dataset = "dataset/train.csv"
    test_dataset= "dataset/test.csv"
    vector_file = open(f"glov.6B/glove.6B.{args.dimension}d.txt", encoding = "utf-8")

    X_train_avg = []
    X_test_avg = []

    textclassifier = EmojiTextClassifier(dim = args.dimension)
    X_train, Y_train = textclassifier.load_dataset(train_dataset)
    X_test, Y_test = textclassifier.load_dataset(test_dataset)

    textclassifier.load_feature_vector(vector_file)

    for x_train in X_train:
        X_train_avg.append(textclassifier.sentence_to_feature_vectors_avg(x_train))
        
    X_train_avg = np.array(X_train_avg)


    for x_test in X_test:
        X_test_avg.append(textclassifier.sentence_to_feature_vectors_avg(x_test))

    X_test_avg = np.array(X_test_avg)

    
    Y_train_one_hot = tf.keras.utils.to_categorical(Y_train, num_classes = 5)
    Y_test_one_hot = tf.keras.utils.to_categorical(Y_test, num_classes = 5)

   
    textclassifier.load_model()
    output = textclassifier.train(X_train_avg, Y_train_one_hot)
    plt.plot(output.history["accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Train accuracy {args.dimension}d - Dropout")
    plt.savefig("output/accuracy-{args.dimension}-Dropout.png")
    plt.show()

    plt.plot(output.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Train loss {args.dimension}d - Dropout")
    plt.savefig(f"output/loss-{args.dimension}-Dropout.png")
    plt.show()


    result = textclassifier.test(X_test_avg, Y_test_one_hot)
    print(f"Test loss : ",result[0], ",Test accuracy : ",result[1])


    #Inference
    
    start = time.time()
    for i in range(100):
        result = textclassifier.inference(args.sentence)
        
    inference_avg_time = (time.time() - start) / 100 

    print(result) 
    print("Inference average time : ", inference_avg_time)



