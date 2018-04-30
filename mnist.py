# code inspiration is from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# this code solves recognition of handwritten numbers with neural network.
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import utils
from keras.datasets import mnist

def clear_console():
    clear = "\n" * 100
    print(clear)

def show_image(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def to_percentage(number):
    return round(number,2)*100

def preprocess_data(train_images, test_images):
    # Reshape data to 1 color channel
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)

    # set float type
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # normalize data to range [0,1]
    train_images /= 255
    test_images /= 255

    return train_images, test_images

def preprocess_labels(train_labels, test_labels):
    train_labels = utils.to_categorical(train_labels, 10)
    test_labels = utils.to_categorical(test_labels, 10)

    return train_labels, test_labels

def main():
    # download and load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # prepare data for neural network
    train_images, test_images = preprocess_data(train_images, test_images)
    train_labels, test_labels = preprocess_labels(train_labels, test_labels)

    # define Sequential model
    network_model = Sequential()

    # add Convolution layers
    network_model.add(Convolution2D(32, 3, strides=3, activation='relu', input_shape=(1,28,28),  padding="same"))
    network_model.add(Convolution2D(32, 3, strides=3, activation='relu',  padding="same"))

    # add Pooling to reduce parameters
    network_model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

    # add Dropout to prevent overfitting
    network_model.add(Dropout(0.25))

    # Flatten input to single number
    network_model.add(Flatten())

    # add Dense layers => neuron to neuron
    network_model.add(Dense(128, activation='relu'))
    network_model.add(Dropout(0.5))
    network_model.add(Dense(10, activation='softmax'))

    # compile our network model
    network_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('\nTeaching our neural network started!\n')
    starting_time = time.time()

    # train our network model
    network_model.fit(train_images, train_labels, batch_size=32, verbose=1, epochs=10)
    
    ending_time = time.time()
    teaching_time = round((ending_time - starting_time),2)
    print('\nTeaching of neural network took {} seconds.\n'.format(teaching_time))

    # evaluate test data on trained model
    score = network_model.evaluate(test_images, test_labels, verbose=0)
    
    success_percentage = to_percentage(score[1])
    print('Neural network predicts with {}% success.\n'.format(success_percentage))

    # make a prediction
    start_index = np.random.randint(0,100)
    end_index = start_index+1

    # show digit to classification
    show_image(test_images[start_index])

    # print digit to classification
    input_number = np.argmax(test_labels[start_index:end_index])
    print('Digit to classification is: {} \n'.format(input_number))

    # prediction
    prediction = network_model.predict(test_images[start_index:end_index]) 

    # print classified digit
    predicted_number = np.argmax(prediction)
    print('Neural network classified the number as digit: {}\n'.format(predicted_number))


if __name__ == "__main__":
    try:
        clear_console()
        main()
    finally:
        print('Exit of program :)')
