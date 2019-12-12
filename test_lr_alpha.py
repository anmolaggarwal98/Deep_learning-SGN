
# imports from own modules
from model_lr_vs_alpha import model_lr
from mnist import mnist2
from fashion_mnist import fashion_mnist
from findclosest import findclosest
from estimate_alpha import estimate_alpha_mohammed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

'''This file runs the model_lr from model_lr_vs_alpha.py on MNIST data_set
and saves the alphas (i.e. the output we get) to a file in MNIST_activation folder
as a pickle file
Output: it plots the graph of learning_rate against alpha'''

start = time.perf_counter()

trX, trY, teX, teY = mnist2()   #outputs the data flattened and normalized
num_epochs = 3
learning_rate_size = 3

estimate_alpha_train, estimate_alpha_test = model_lr(trX, trY, teX, teY,
                                                        num_epochs = num_epochs,
                                                        learning_rate_size = learning_rate_size,
                                                        batch_size = 500,
                                                        activation = 'relu',
                                                        input_dim = 784,
                                                        neurons_hidden_layer = 100,
                                                        output_dim = 10)

# save the output in the right directory using pickle format
# make sure you create an empty folder "MNIST_activation" before running this

print("make sure you create an empty folder 'MNIST_lr' before running this or else you will get an error msg\n")
pickle_out = open(f'MNIST_lr/relu_trial_alpha_train_lr{learning_rate_size}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_train,pickle_out)
pickle_out.close()

pickle_out = open(f'MNIST_lr/relu_trial_alpha_test_lr{learning_rate_size}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_test,pickle_out)
pickle_out.close()

estimate_alpha_train = pickle.load(open(f'MNIST_lr/relu_trial_alpha_train_lr{learning_rate_size}.pickle','rb'))
estimate_alpha_test = pickle.load(open(f'MNIST_lr/relu_trial_alpha_test_lr{learning_rate_size}.pickle','rb'))

mean_train_alpha = np.mean(estimate_alpha_train,axis=1)
mean_test_alpha = np.mean(estimate_alpha_test,axis=1)
learning_rate_list = np.linspace(0.001,0.1,learning_rate_size)

print('\nBuilding Graph....')
plt.plot(learning_rate_list,mean_train_alpha,'--.r',linewidth = 1,label='train');
plt.plot(learning_rate_list,mean_test_alpha,'--.b',linewidth = 1,label='test');
plt.xlabel(r'Learning Rate $\eta$');
plt.ylabel(r'Estimate $\hat{\alpha}$');
plt.grid();
plt.legend();
plt.title('MNIST: Relu');
#plt.ylim([0.8,1.5])
plt.savefig(f'MNIST_lr/relu_trial_graph_lr{learning_rate_size}.png', format='png', dpi=1200)
plt.show()

end = time.perf_counter()
print(f'Wall Time: {round(end-start,3)} secs \n')
