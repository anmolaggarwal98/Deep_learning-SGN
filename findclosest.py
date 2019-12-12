import numpy as np

def findclosest(K):
    '''Goal of this code is to find two integers K1, K2 such that
    K = K1 x K2 and also that K1 is the closest divisor to sqrt(K)

    This code is essentially used for calculating K1 and K2 for the
    Mohammadi 2015 Theorem 2 in my paper given I know the value K'''

    sqrt = K**0.5

    def factors(n):
        '''Calculates all the factor of number n'''
        return set(
            factor for i in range(1, int(n**0.5) + 1) if n % i == 0
            for factor in (i, n//i)
        )

    list1 = list(factors(K))
    list1 = np.array(list1)
    difference = abs(sqrt - list1)

    K1 = list1[np.argmin(difference)]
    K2 = int(K/K1)
    return K1,K2

if __name__ == '__main__':
    #testing if things work
    K = 27000
    print(findclosest(K))  #expected K1 = 150, K2 = 180
