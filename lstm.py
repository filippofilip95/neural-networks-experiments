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

cls = lambda: print('\n'*100)
cls()

embed_dim = 128
lstm_out = 200
batch_size = 32
maxlen= 900

def get_reviews_from_file(how_many = 6000):
    # extract archive from https://www.yelp.com/dataset/download to acces ./review.json (filesize is 4,21 GB)
    with open('./review.json','r') as review_json:
        data = {
            "stars": [],
            "text": []
        }

        replace_unwanted = lambda x: re.sub('([^a-zA-z0-9\s])|(\n)','',x)

        for i, row in enumerate(review_json):
            row = json.loads(row)
            
            data['stars'].append(int(row['stars']))
            data['text'].append(replace_unwanted(row['text']))

            if(i==how_many-1):
                break

        return data


def preprocess_data(data):
    data['sentiment'] = ['pos' if (x>3) else 'neg' for x in data['stars']]

    tokenizer = Tokenizer(num_words=2500, lower=True, split=' ' )
    tokenizer.fit_on_texts(data['text'])

    X = tokenizer.texts_to_sequences(data['text'])
    X = pad_sequences(X, maxlen=maxlen)

    return X

def build_lstm_model(X):

    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length = X.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_out))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def make_train_test_data(X):
    Y = pd.get_dummies(data['sentiment'])
    return train_test_split(X,Y, test_size=0.20, random_state=36)

def humanize_prediction(predicate):
    print('\nOutput of neural network is:')
    print(predicate)
    print('\nNow output for humas is:')
    for i, prediction in enumerate(np.argmax(predicate,1)):
        if(prediction==0):
            print('Review with index {} is postive'.format(i))
        else:
            print('Review with index {} is negative'.format(i))


if __name__ == "__main__":
    try:
        # prepocess data
        data = get_reviews_from_file()
        X = preprocess_data(data)
        model = build_lstm_model(X)
        X_train, X_valid, Y_train, Y_valid = make_train_test_data(X)

        print('\nTeaching our neural network started!\n')
        starting_time = time.time()

        # train  model
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=2,  verbose=1)

        ending_time = time.time()
        teaching_time = round((ending_time - starting_time),2)
        print('\nTeaching of neural netwrok took {} seconds.\n'.format(teaching_time))

        # evaluate model
        score, acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
        print("\nAccuracy of model is: %.2f" % (acc))

        #make a prediction
        test_reviews = {
            "stars": [2,5],
            "text": ['This was very disappointing.  We had a reservation and checked in right on time.  They had at least 10 tables that were open but they made us wait 25 minutes.  We reminded them twice we were there.  The service continued to be disappointing. We had to wait to get water and even then they only gave 2 out of the 4 of us water.  Took at least 20 min to get bread and to have a waiter come over.  The food was good but not phenomenal.  My steak was luke warm.  They should have a lot better service and quality for the prices you pay.  Mastro, Dominick and Steak 44 are a lot better and 2 of those are not very far from this location.  We would not eat here again.','These guys are great . My engine was totally messed up and I just needed an oil change to keep it going so I could take it to mechanic that recently fixed it and they hooked me up . I highly recommend this location as they are honest and fair']
        }
        test_reviews = preprocess_data(test_reviews)

        print('\nMaking prediction with 2 reviews.')
        predicate = model.predict(test_reviews, batch_size=None, verbose=0, steps=None)
        
        humanize_prediction(predicate)
  
    finally:
        print('\nExit of program :)')
    