
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

import argparse
import sys

'''This file runs the model_lr from model_lr_vs_alpha.py on MNIST data_set
and saves the alphas (i.e. the output we get) to a file in MNIST_activation folder
as a pickle file
Output: it plots the graph of learning_rate against alpha'''

start = time.perf_counter()

#trX, trY, teX, teY = mnist2()   #outputs the data flattened and normalized
num_epochs = 3
learning_rate_size = 3
dataset_dict = {'mnist': mnist2(), 'fashion_mnist': fashion_mnist()}

# this parser stuff is so that I can make changes in the command line if I want to
parser = argparse.ArgumentParser()  #this is an object of a class ArgumentParser
parser.add_argument('--dataset',type=str,default='mnist',help='dataset we choose from (mnist,fashion_mnist)')
parser.add_argument('--epochs',type=int, default=num_epochs,help='number of times you need to iterate over the whole function')
parser.add_argument('--learning_rate_size',type=int,default=learning_rate_size,help='number of learning_rate you want to test against from 0.001 to 0.1')
parser.add_argument('--batch_size',type=int,default=500,help='batch size')
parser.add_argument('--activation',type=str,default='relu',help='activation function used (relu,sigmoid)')
parser.add_argument('--input_dim',type=int,default=784,help='dimension of the input layer which is 28*28')
parser.add_argument('--neurons_hidden_layer',type=int,default=100,help='number of neurons in the hidden layer')
parser.add_argument('--output_dim',type=int,default=10,help='size of the output class which is 0,....,9 i.e 10')
args = parser.parse_args()  #creates an object called namespace which is sort of like a dict vars(args)

print("\nWhen running this from commandline\ncd 'OneDrive - Nexus365\Oxford Masters\Modules\Theories of Deep Learning\Project\Code\Mini_project_code'")
print('command: `python test_epoch_alpha_relu.py -h`')
print('command: `python test_epoch_alpha_relu.py --epochs=50 --activation=sigmoid --learn_rate=0.01`\n')

trX, trY, teX, teY = dataset_dict[args.dataset]   #outputs the data flattened and normalized

estimate_alpha_train, estimate_alpha_test = model_lr(trX, trY, teX, teY,
                                                        num_epochs = args.epochs,
                                                        learning_rate_size = args.learning_rate_size,
                                                        batch_size = args.batch_size,
                                                        activation = args.activation,
                                                        input_dim = args.input_dim,
                                                        neurons_hidden_layer =args.neurons_hidden_layer,
                                                        output_dim = args.output_dim)

# save the output in the right directory using pickle format
# make sure you create an empty folder "MNIST_activation" before running this

print(f"\nmake sure you create an empty folder with the exact name '{args.dataset}_lr' before running this or else you will get an error msg\n")
pickle_out = open(f'{args.dataset}_lr/{args.activation}_trial_alpha_train_lr{args.learning_rate_size}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_train,pickle_out)
pickle_out.close()

pickle_out = open(f'{args.dataset}_lr/{args.activation}_trial_alpha_test_lr{args.learning_rate_size}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_test,pickle_out)
pickle_out.close()

estimate_alpha_train = pickle.load(open(f'{args.dataset}_lr/{args.activation}_trial_alpha_train_lr{args.learning_rate_size}.pickle','rb'))
estimate_alpha_test = pickle.load(open(f'{args.dataset}_lr/{args.activation}_trial_alpha_test_lr{args.learning_rate_size}.pickle','rb'))

mean_train_alpha = np.mean(estimate_alpha_train,axis=1)
mean_test_alpha = np.mean(estimate_alpha_test,axis=1)
learning_rate_list = np.linspace(0.001,0.1,args.learning_rate_size)

end = time.perf_counter()
print(f'Wall Time: {round(end-start,3)} secs \n')

print('\nBuilding Graph....')
plt.plot(learning_rate_list,mean_train_alpha,'--.r',linewidth = 1,label='train');
plt.plot(learning_rate_list,mean_test_alpha,'--.b',linewidth = 1,label='test');
plt.xlabel(r'Learning Rate $\eta$');
plt.ylabel(r'Estimate $\hat{\alpha}$');
plt.grid();
plt.legend();
plt.title(f'{args.dataset}: {args.activation}');
#plt.ylim([0.8,1.5])
plt.savefig(f'{args.dataset}_lr/{args.activation}_trial_graph_lr{args.learning_rate_size}.png', format='png', dpi=1200)
plt.show()
