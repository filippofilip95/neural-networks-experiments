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
from keras.utils import plot_model

def clear_console():
    clear = "\n" * 100
    print(clear)

def show_image(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    fig = plt.figure()
    plt.legend([None])
    plt.imshow(pixels, cmap='gray')
    fig.savefig('./src/images/mnist_digit_to_predicate.png')


def show_learning_rate_graphs(history):
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig('./src/images/mnist_acc_history.png')

    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig('./src/images/mnist_loss_history.png')

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

def create_and_compile_model():
    # define Sequential model
    network_model = Sequential()

    # add Convolution layer
    network_model.add(Convolution2D(30, (5,5), activation='relu', input_shape=(1,28,28),  padding="same"))

    # add Pooling to reduce parameters
    network_model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

    # add Convolution layer
    network_model.add(Convolution2D(15, (3,3), activation='relu', padding="same"))

    # add Dropout to prevent overfitting
    network_model.add(Dropout(0.25))

    # Flatten input to single number
    network_model.add(Flatten())

    # add Dense layers => neuron to neuron
    network_model.add(Dense(128, activation='relu'))
    network_model.add(Dense(10, activation='softmax'))

    # compile our network model
    network_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # save model visualization to file
    plot_model(network_model, to_file='./src/images/mnist_model.png', show_shapes=True)

    return network_model


def main():
    # download and load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # prepare data for neural network
    train_images, test_images = preprocess_data(train_images, test_images)
    train_labels, test_labels = preprocess_labels(train_labels, test_labels)

    network_model = create_and_compile_model()    

    print('\nTeaching of neural network started!\n')
    starting_time = time.time()

    # train our network model
    history = network_model.fit(train_images, train_labels, batch_size=200, verbose=1, epochs=10, validation_data=(test_images,test_labels))
    show_learning_rate_graphs(history)

    ending_time = time.time()
    teaching_time = round((ending_time - starting_time),2)
    print('\nTeaching of neural network took {} seconds.\n'.format(teaching_time))

    # evaluate test data on trained model
    score = network_model.evaluate(test_images, test_labels, verbose=0)
    print('Neural network predicts with {}% success.\n'.format(score[1]*100))

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
