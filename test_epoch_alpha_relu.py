
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

import argparse
import sys

'''This file runs the model_alpha from model_epoch_vs_alpha.py on MNIST data_set
and saves the alphas (i.e. the output we get) to a file in MNIST_activation folder
as a pickle file
Output: it plots the graph of epochs against alpha'''

start = time.perf_counter()

num_epochs = 50 #setting epochs default to be this
dataset_dict = {'mnist': mnist2(), 'fashion_mnist': fashion_mnist()}

# this parser stuff is so that I can make changes in the command line if I want to
parser = argparse.ArgumentParser()  #this is an object of a class ArgumentParser
parser.add_argument('--dataset',type=str,default='mnist',help='dataset we choose from (mnist,fashion_mnist)')
parser.add_argument('--epochs',type=int, default=num_epochs,help='number of times you need to iterate over the whole function')
parser.add_argument('--learn_rate',type=float,default=0.1,help='learning rate is your step size')
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

estimate_alpha_train, estimate_alpha_test = model_alpha(trX, trY, teX,teY,num_epochs = args.epochs,
                                                        learn_rate = args.learn_rate,
                                                        batch_size = args.batch_size,
                                                        activation =args.activation,
                                                        input_dim = args.input_dim,
                                                        neurons_hidden_layer =args.neurons_hidden_layer,
                                                        output_dim = args.output_dim)

# save the output in the right directory using pickle format
# make sure you create an empty folder "MNIST_activation" before running this
print(f"make sure you create an empty folder with exact name '{args.dataset}_activation' before running this or else you will get an error msg\n")
pickle_out = open(f'{args.dataset}_activation/{args.activation}_trial_Estimate_alpha_train_epoch{args.epochs}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_train,pickle_out)
pickle_out.close()

pickle_out = open(f'{args.dataset}_activation/{args.activation}_trial_Estimate_alpha_test_epoch{args.epochs}.pickle','wb')  #create a file X.pickle
pickle.dump(estimate_alpha_test,pickle_out)
pickle_out.close()

Epoch = np.arange(1,args.epochs+1) #creates an array for plotting

#loading the saved data
estimate_alpha_train = pickle.load(open(f'{args.dataset}_activation/{args.activation}_trial_Estimate_alpha_train_epoch{args.epochs}.pickle','rb'))
estimate_alpha_test = pickle.load(open(f'{args.dataset}_activation/{args.activation}_trial_Estimate_alpha_test_epoch{args.epochs}.pickle','rb'))

end = time.perf_counter()

print(f'Wall Time: {round(end-start,3)} secs \n')

print('\nBuilding Graph.....')
plt.plot(Epoch,estimate_alpha_train,'--.r',linewidth = 1,label='train');
plt.plot(Epoch,estimate_alpha_test,'--.b',linewidth = 1,label='test');
plt.xlabel(r'Epoch');
plt.ylabel(r'Estimate $(\hat{\alpha})$');
plt.grid()
plt.legend()
plt.title(f'{args.dataset}: {args.activation}')
plt.savefig(f'{args.dataset}_activation/{args.activation}_graph_epoch{args.epochs}.png', format='png', dpi=1200)
plt.show()
