# code inspiration is from https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
# this code solves sentiment classification with LMNST neural network
import re
import json
import time
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# clear console
cls = lambda: print('\n'*100)
cls()

# global variables
embed_dim = 128
lstm_out = 200
batch_size = 32
maxlen= 900

def get_reviews_from_file(how_many = 6000):
    # extract archive from https://www.yelp.com/dataset/download to access ./review.json (filesize is 4,21 GB)
    with open('./review.json','r') as review_json:
        # init shape of dict, we need only text and stars from reviews file
        data = {
            "stars": [],
            "text": []
        }
        # this speaks for itself
        replace_unwanted = lambda x: re.sub('([^a-zA-z0-9\s])|(\n)','',x)

        for i, row in enumerate(review_json):
            row = json.loads(row)
            
            # fill our dict (data) with parsed values from file 
            data['stars'].append(int(row['stars']))
            data['text'].append(replace_unwanted(row['text']))

            # break cycle if we reached specified value of reviews
            if(i==how_many-1):
                break

        return data


def preprocess_data(data):
    # add sentiment attribute for each review in dict
    # if review has more than 3 stars, we consider this review as positive, on the other hand review is negative
    data['sentiment'] = ['pos' if (x>3) else 'neg' for x in data['stars']]

    # init tokenizer, keep only 2500 most common words, lowerize sentences, split sentences by space
    tokenizer = Tokenizer(num_words=2500, lower=True, split=' ' )
    # fill tokenizer with our texts
    tokenizer.fit_on_texts(data['text'])
    # transform texts to sequence of integers
    X = tokenizer.texts_to_sequences(data['text'])
    # transform each sequence to same length (specified by maxlen)
    X = pad_sequences(X, maxlen=maxlen)

    return X

def build_lstm_model(X):
    # init sequential model
    model = Sequential()
    # vectorize input
    # input_dim (size of vocabulary) is 2500 because we have only 2500 most common words (line 53)
    # embed_dim (shape of output)
    # input_length (length of input sequences), just get shape of first value because lenghts of all values are same (line 59), also could be replaced with maxlen value
    model.add(Embedding(2500, embed_dim, input_length=X.shape[1]))
    # prevent from overfitting
    model.add(Dropout(0.2))
    # add LSTM layer with specified shape of output
    model.add(LSTM(lstm_out))
    # prevent from overfitting
    model.add(Dropout(0.2))
    # add Dense layer with specified shape of output, there are only 2 outputs, because review can be only positive or negative
    model.add(Dense(2, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def make_train_test_data(X):
    # convert categorical variables to indicator variables
    # example input:
    # [ a , b ]
    # example output:
    #  a  b
    #  1  0
    #  0  1    
    Y = pd.get_dummies(data['sentiment'])
    # split data to test,train and their validation
    return train_test_split(X,Y, test_size=0.20, random_state=36)

def humanize_prediction(predicate):
    # print output of predictions for robots :) e.g. for Sophia
    print('\nOutput of neural network is:')
    print(predicate)
    # print formated output for humans :)
    print('\nNow output for humans is:')
    indexes_of_max_values = np.argmax(predicate,1)
    for i, prediction in enumerate(indexes_of_max_values):
        result = 'positive'
        if(prediction!=0):
            result = 'negative'
        print('Review with index {} is {}'.format(i,result))

if __name__ == "__main__":
    try:
        # preprocess data, methods are described above
        data = get_reviews_from_file()
        X = preprocess_data(data)
        model = build_lstm_model(X)
        X_train, X_valid, Y_train, Y_valid = make_train_test_data(X)

        print('\nTeaching our neural network started!\n')
        starting_time = time.time()

        # train  model with inputs and their validations
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=3,  verbose=1)

        ending_time = time.time()
        teaching_time = round((ending_time - starting_time),2)
        print('\nTeaching of neural netwrok took {} seconds.\n'.format(teaching_time))

        # evaluate model on test data and print accuracy
        score, acc = model.evaluate(X_valid, Y_valid, verbose=2, batch_size=batch_size)
        print("\nAccuracy of model is: %.2f" % (acc))

        # make a prediction
        # initialize 2 sentences for prediction
        test_reviews = {
            "stars": [2,5],
            "text": ['This was very disappointing.  We had a reservation and checked in right on time.  They had at least 10 tables that were open but they made us wait 25 minutes.  We reminded them twice we were there.  The service continued to be disappointing. We had to wait to get water and even then they only gave 2 out of the 4 of us water.  Took at least 20 min to get bread and to have a waiter come over.  The food was good but not phenomenal.  My steak was luke warm.  They should have a lot better service and quality for the prices you pay.  Mastro, Dominick and Steak 44 are a lot better and 2 of those are not very far from this location.  We would not eat here again.','These guys are great . My engine was totally messed up and I just needed an oil change to keep it going so I could take it to mechanic that recently fixed it and they hooked me up . I highly recommend this location as they are honest and fair']
        }
        # preprocess sentences for prediction
        test_reviews = preprocess_data(test_reviews)

        # make prediction of sentiment for our 2 sentences 
        print('\nMaking prediction with 2 reviews.')
        predicate = model.predict(test_reviews, batch_size=None, verbose=0, steps=None)
        
        # we want to view some output for humans :) we aren't robots 
        humanize_prediction(predicate)
  
    finally:
        print('\nExit of program :)')
    