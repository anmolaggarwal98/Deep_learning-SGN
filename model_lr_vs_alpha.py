from mnist import mnist2
from fashion_mnist import fashion_mnist
from estimate_alpha import estimate_alpha_mohammed
from model_epoch_vs_alpha import feed_forward,grads
from estimate_alpha import estimate_alpha_mohammed

import numpy as np
import time

def model_lr(trX, trY, teX, teY,
            num_epochs = 100,
            learning_rate_size = 50,
            batch_size = 500,
            activation = 'relu',
            input_dim = 784,
            neurons_hidden_layer = 100,
            output_dim = 10):

    '''This model calculates the values of tail index i.e. alpha for both the
    training images and also the test images by passing in through forward_feed
    and then uses backgropagation and stochastic_gradient_descent. Then it varies the
    value of learning rate (note in `model_alpha that is fixed at 0.1`) and see how
    tail index changes. The graph of this can be seen on `test_lr_alpha.py`

    Parameter:
     - trX, trY, teX, teY = MNIST/fashion_mnist data after flatten and normalized
     - num_epochs: int
                   Provide the Epoch i.e. number of time to iterate through WHOLE data_set
     - learning_rate_size: int
                    It splits the learning_rate_list which goes from 0.001 to 0.1 into int number of
                    equispaced points. By default it is 50.
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
    # calculates two weight matrices
    weights = [np.random.randn(*w) * 0.1 for w in [(input_dim, neurons_hidden_layer), (neurons_hidden_layer, output_dim)]] #creates a tensor

    learning_rate_list = np.linspace(0.001,0.1,learning_rate_size)  #outputs equispaced lr
    train_length = len(trX)

    estimate_alpha_train = np.zeros((learning_rate_size,num_epochs)) #zero matrix to store alphas
    estimate_alpha_test = np.zeros((learning_rate_size,num_epochs)) #same for test alphas
    count = 0

    print(f'Calculation in progress. Please be patient while it gives you your intended results\nUsing {activation} as the activation function\n')


    for index,learn_rate in enumerate(learning_rate_list):
        count+=1
        print(f'Iter: {count}')
        print(f"learning_rate: {learn_rate}")

        for k in range(num_epochs):

            #calculating overall gradient jacobian matrix for all dataset
            GD_gradient = grads(trX,trY,weights,activation)[1]
            GD_gradient = GD_gradient.reshape(-1,neurons_hidden_layer*output_dim)[0]

            each_minibatch_gradients = []  #for each epoch to store all training stochastic gradient
            each_test_minibatch_gradients = [] #same as above for test

            # this loop for calculating stochastic gradient Noise for training data
            for j in range(0, len(trX), batch_size):
                X_batch, Y_batch = trX[j:j+batch_size], trY[j:j+batch_size]

                both_weights_gradients = grads(X_batch, Y_batch, weights,activation)

                batch_gradients = both_weights_gradients[1] #100x10 #old_weights
                each_minibatch_gradients.append(batch_gradients.reshape(-1,neurons_hidden_layer*output_dim)[0]) #after flattening

                weights -= learn_rate * both_weights_gradients  #updating both the weights in one go

            stochastic_gradient_noise = GD_gradient - each_minibatch_gradients #stochastic noise
            stochastic_gradient_noise = stochastic_gradient_noise.reshape(-1) #flattening it to a single vector K=hidden_neuronxoutput_dimx(ceiling(n/b))

            #same as above but for test images
            for j in range(0,len(teX),batch_size):
                X_test_batch, Y_test_batch = teX[j:j+batch_size], teY[j:j+batch_size]

                both_test_weights_gradients = grads(X_test_batch, Y_test_batch, weights,activation)

                batch_test_gradients = both_test_weights_gradients[1] #100x10 #old_weights
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

            estimate_alpha_train[index][k] = alpha_k
            estimate_alpha_test[index][k] = alpha_test_k

        print(f'Alpha_train: {estimate_alpha_train[index]}')
        print(f'Alpha_test: {estimate_alpha_test[index]}\n')

    print(f'Estimate alpha mean train per learning rate: \n{np.mean(estimate_alpha_train,axis=1)}')
    print('')
    print(f'Estimate alpha mean test per learning rate: \n{np.mean(estimate_alpha_test, axis = 1)}')


    return estimate_alpha_train, estimate_alpha_test

if __name__=='__main__':
    start = time.perf_counter()

    trX, trY, teX, teY = mnist2()   #outputs the data flattened and normalized


    activation = 'relu'

    estimate_alpha_train, estimate_alpha_test = model_lr(trX, trY, teX, teY,
                                                            num_epochs = 3,
                                                            learning_rate_size = 10,
                                                            batch_size = 500,
                                                            activation = activation,
                                                            input_dim = 784,
                                                            neurons_hidden_layer = 100,
                                                            output_dim = 10)

    end = time.perf_counter()
    print(f'Wall Time: {round(end-start,3)} secs \n')
