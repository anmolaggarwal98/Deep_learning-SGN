import numpy as np
import tensorflow as tf

def mnist2():
    '''This function imports the mnist data_set from
    keras and filters it into the format I need which is:
     - It splits the data into training and testing splitting
     it into 60,000 and 10,0000 images respectively
     - Each image is 28x28 so it flattens the image out into
     a single vector of dimension 784 (28x28)
     - Since each pixel (i.e. value of each entry in image matrix)
     can range from 0 to 255, the code normalizes it
     - It also does onehotcoding of the training and testing labels'''

    mnist = tf.keras.datasets.mnist #28x28 images of handwritten digits 0-9

    #splitting into its different components
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    #normalizing and flattening
    x_train = x_train.reshape(-1,28*28)/255.0
    x_test = x_test.reshape(-1,28*28)/255.0

    def onehot(integer_labels):
            #Return matrix whose rows are onehot encodings of integers.
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehotL = np.zeros((n_rows, n_cols), dtype='uint8')
            onehotL[np.arange(n_rows), integer_labels] = 1
            return onehotL

    y_train = onehot(y_train)
    y_test = onehot(y_test)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    #testing if things work
    train_images, train_labels, test_images, test_labels = mnist2()
    print(train_images[0])
