
# imports from own library
from mnist import mnist2
from fashion_mnist import fashion_mnist
from estimate_alpha import estimate_alpha_mohammed

import numpy as np
import time

# calculate sigmoid function and its derivative
sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda y: y * (1 - y)

def feed_forward(X, weights, activation):
    '''Feed_forward in the net with X being the input to the neuron '''
    a = [X]
    for w in weights:
        if activation == 'sigmoid':
            a.append(sigmoid(a[-1].dot(w)))     # Using sigmoid
        elif activation == 'relu':
            a.append(np.maximum(a[-1].dot(w),0)) # Using ReLU
        else:
            print("The activation function should either be 'sigmoid' or 'relu'. Please adjust accordingly")
    return a

#####################################################################################################

def grads(X, Y, weights,activation):
    '''Calculates the Jacobian matrix for each iteration'''
    grads = np.empty_like(weights)
    a = feed_forward(X, weights,activation)
    delta = a[-1] - Y
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1):
        if activation == 'sigmoid':
            delta = np.dot(delta, weights[i].T) * d_sigmoid(a[i]) # Using Sigmoid
        elif activation == 'relu':
            delta = (a[i] > 0) * delta.dot(weights[i].T) #Grad of ReLU exist for x>0
        else:
            print("The activation function should either be 'sigmoid' or 'relu'. Please adjust accordingly")
        grads[i-1] = a[i-1].T.dot(delta)
    return grads / len(X)

######################################################################################################

def model_alpha(trX, trY, teX, teY, num_epochs = 10, learn_rate = 0.1, batch_size = 500,activation = 'sigmoid',
                input_dim = 784,neurons_hidden_layer = 100, output_dim = 10):

    '''This model calculates the values of tail index i.e. alpha for both the
    training images and also the test images by passing in through forward_feed
    and then uses backgropagation and stochastic_gradient_descent

    Parameter:
     - trX, trY, teX, teY = MNIST/fashion_mnist data
     - num_epochs: int
                   Provide the Epoch i.e. number of time to iterate through WHOLE data_set
     - learn_rate: float
                    Step size when propagating back through the network, usually takes values
                    between 0.00001 and 1
     - batch_size: int
                   when doing stochastic_gradient_descent, what mini_batch size you need
     - activation = str
                  lets you use either sigmoid or relu function and you tell the code as a string
     - input_dim: int
                  neurons in input layer, by default it is 784 (28x28)
     - neurons_hidden_layer = int
                  neurons in the hidden layer, by default 100
     - output_dim = int
                  neurons in the output layer, by default it is 10 because I am working
                  with MNIST data which gives output in one of the 10 classes (0,1,...,9)

'''

    print('Calculation in progress. Please be patient while it gives you your intended results')
    print(f'Using {activation} as the activation function activation')
    print('Model prints results every 5 epochs\n')

    # calculates two weight matrices
    weights = [np.random.randn(*w) * 0.1 for w in [(input_dim, neurons_hidden_layer), (neurons_hidden_layer, output_dim)]] #creates a tensor

    estimate_alpha_train = np.zeros(num_epochs) #empty list to store alpha from train
    estimate_alpha_test = np.zeros(num_epochs)  #to store for test
    train_length = len(trX)

    for k in range(num_epochs):
        if (k+1)%5==0:  #prints every 5 iterations
            print(f'Epoch:{k+1}')

        #calculating overall gradient jacobian matrix for all dataset
        GD_gradient = grads(trX,trY,weights, activation)[1]
        GD_gradient = GD_gradient.reshape(-1,neurons_hidden_layer*output_dim)[0]

        each_minibatch_gradients = []  #for each epoch to store all training stochastic gradient
        each_test_minibatch_gradients = [] #same as above for test

        # this loop for calculating stochastic gradient Noise for training data
        for j in range(0, len(trX), batch_size):
            X_batch, Y_batch = trX[j:j+batch_size], trY[j:j+batch_size]

            both_weights_gradients = grads(X_batch, Y_batch, weights,activation = activation)

            batch_gradients = both_weights_gradients[1] #100x10
            each_minibatch_gradients.append(batch_gradients.reshape(-1,neurons_hidden_layer*output_dim)[0]) #after flattening

            weights -= learn_rate * both_weights_gradients  #updating both the weights in one go

        stochastic_gradient_noise = GD_gradient - each_minibatch_gradients  #stochastic noise
        stochastic_gradient_noise = stochastic_gradient_noise.reshape(-1) #flattening it to a single vector K=hidden_neuronxoutput_dimx(ceiling(n/b))

        ###################################

        #same as above but for test images
        for j in range(0,len(teX),batch_size):
            X_test_batch, Y_test_batch = teX[j:j+batch_size], teY[j:j+batch_size]

            both_test_weights_gradients = grads(X_test_batch, Y_test_batch, weights, activation)

            batch_test_gradients = both_test_weights_gradients[1] #100x10
            each_test_minibatch_gradients.append(batch_test_gradients.reshape(-1,neurons_hidden_layer*output_dim)[0]) #after flattening

        stochastic_test_gradient_noise = GD_gradient - each_test_minibatch_gradients
        stochastic_test_gradient_noise = stochastic_test_gradient_noise.reshape(-1) #flattening it to a single vector K=hidden_neuronxoutput_dimx(ceiling(n/b))

        #calculating the alpha using Mohammadi estimator
        # doing try except to remove dividing by zero error
        try:
            alpha_k = estimate_alpha_mohammed(stochastic_gradient_noise)
            alpha_test_k = estimate_alpha_mohammed(stochastic_test_gradient_noise)

        except Exception as e:
            #print('log error occured')
            alpha_k = estimate_alpha[k-1]
            alpha_test_k = estimate_alpha_test[k-1]

        # storing in this vector
        estimate_alpha_train[k] = alpha_k
        estimate_alpha_test[k] = alpha_test_k

        if (k+1)%5==0:
            print(f'alpha_train{k+1}: {alpha_k}')
            print(f'alpha_test{k+1}: {alpha_test_k}')

            # testing on test images
            prediction = np.argmax(feed_forward(teX, weights,activation=activation)[-1], axis=1)
            # calculating accuracy
            print ('Accuracy',f'{100*np.mean(prediction == np.argmax(teY, axis=1))} %\n')

    return estimate_alpha_train, estimate_alpha_test

##########################################################################################

if __name__=='__main__':
    start = time.perf_counter()

    trX, trY, teX, teY = mnist2()   #outputs the data flattened and normalized

    activation = 'sigmoid' #'relu'

    estimate_alpha_train, estimate_alpha_test = model_alpha(trX, trY, teX, teY,
                                                            num_epochs = 10,
                                                            learn_rate = 0.1,
                                                            batch_size = 500,
                                                            activation = activation,
                                                            input_dim = 784,
                                                            neurons_hidden_layer = 100,
                                                            output_dim = 10)

    end = time.perf_counter()
    print(f'Wall Time: {round(end-start,3)} secs \n')
