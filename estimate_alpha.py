from findclosest import findclosest
import numpy as np

def estimate_alpha_mohammed(X):
    '''Using Mohammadi Theorem in my paper, this algorithm estimates the
    tail index in stable distribution alpha to very high precision (see my test.py)'''

    mohammed = lambda Y,X: (1/np.log(abs(K_1)))*((1/K_2)*sum(np.log(abs(Y))) - (1/K)*sum(np.log(abs(X))))

    K = len(X)
    K_1,K_2 = findclosest(K)


    Y = np.zeros(K_2)
    for i in range(K_2):
        Y[i] = sum((X[j+(i-1)*K_1] for j in range(K_1)))
    alpha_k = 1/mohammed(Y,X)
    return alpha_k

if __name__ == '__main__':
    #testing if things work

    X = np.random.normal(loc = 0, scale = 1, size = 100)
    print(estimate_alpha_mohammed(X))  #should give a value close to alpha = 2 which means normal distribution
