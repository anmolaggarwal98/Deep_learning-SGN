
# imports from own modules
from model_epoch_vs_alpha import model_alpha
from mnist import mnist2
from fashion_mnist import fashion_mnist
from findclosest import findclosest
from estimate_alpha import estimate_alpha_mohammed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

'''This file runs the model_alpha from model_epoch_vs_alpha.py on MNIST data_set
and saves the alphas (i.e. the output we get) to a file in MNIST_activation folder
as a pickle file
Output: it plots the graph of epochs against alpha'''

start = time.perf_counter()

trX, trY, teX, teY = mnist2()   #outputs the data flattened and normalized
num_epochs = 50

estimate_alpha_train, estimate_alpha_test = model_alpha(trX, trY, teX, teY,num_epochs = num_epochs,
                                                        learn_rate = 0.1,
                                                        batch_size = 500,
                                                        activation = 'relu',
                                                        input_dim = 784,
                                                        neurons_hidden_layer = 100,
                                                        output_dim = 10)

# save the output in the right directory using pickle format
# make sure you create an empty folder "MNIST_activation" before running this
print("make sure you create an empty folder with exact name 'MNIST_activation' before running this or else you will get an error msg")
pickle_out = open(f'MNIST_activation/sigmoid_trial_Estimate_alpha_train_epoch{num_epochs}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_train,pickle_out)
pickle_out.close()

pickle_out = open(f'MNIST_activation/sigmoid_trial_Estimate_alpha_test_epoch{num_epochs}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_test,pickle_out)
pickle_out.close()

Epoch = np.arange(1,num_epochs+1) #creates an array for plotting

#loading the saved data
estimate_alpha_train = pickle.load(open(f'MNIST_activation/sigmoid_trial_Estimate_alpha_train_epoch{num_epochs}.pickle','rb'))
estimate_alpha_test = pickle.load(open(f'MNIST_activation/sigmoid_trial_Estimate_alpha_test_epoch{num_epochs}.pickle','rb'))

print('\nBuilding Graph.....')
plt.plot(Epoch,estimate_alpha_train,'--.r',linewidth = 1,label='train');
plt.plot(Epoch,estimate_alpha_test,'--.b',linewidth = 1,label='test');
plt.xlabel(r'Epoch');
plt.ylabel(r'Estimate $(\hat{\alpha})$');
plt.grid()
plt.legend()
plt.title('MNIST: Sigmoid')
plt.savefig(f'MNIST_activation/sigmoid_graph_epoch{num_epochs}.png', format='png', dpi=1200)
plt.show()

end = time.perf_counter()
print(f'Wall Time: {round(end-start,3)} secs \n')
