#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
diversity_v1: the computation of diversity, used for the diversity_sampling function in apply_filter_v1.py
the distances of all instances put in accumulative setting, then instance with larger distance is more likely to be selected

class diversitySampling
    def __init__(self, X, pool=[], budget=0),  initialized with data, pool and budget
    def updateCplus(self)
    def computeD(self)

conpute the diversity using Euclidean distance
"""

__author__      = "Oggi Rudovic, Beryl Zhang"
__copyright__   = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"

import numpy as np
import pandas as pd
import random

class diversitySampling():
    def __init__(self, X, pool = [], budget = 0):
        """
        initialize the a diversitySampling object

        :param X: the data in shape [N]
        :param pool: pool of data that is used to get the reference point
        :param budget: the number of samples to select
        """
        # self.X is the data in shape [N*]
        self.X = X
        # self.X2 is a copy of X
        self.X2 = np.copy(X)
        # self.k is the number of samples required
        self.k = budget
        # self.m is the number of samples (total)
        self.m = X.shape[0]
        # second dimension of X, which is feature dimension
        self.d = X.shape[1]
        # empty list used to store the pool of comparing
        self.C = list()
        for i in range(pool.shape[0]):
            self.C.append(pool[i])
        # a list of index
        self.ind2 =list(range(self.m))
        # list to store the index of selected sample
        self.newind = list()

    def updateCplus(self):
        """

        :return: self.C contains the data for comparing
        """
        cnt = 0
        #randomly select a point if no reference point given, randomly select a point and add to selection-pool
        if len(self.C) == 0:
            tmp = random.randint(self.m)
            xel = self.X[tmp]
            self.C.append(xel)
            self.newind.append(self.ind2[tmp])
            self.X2 = np.delete(self.X2,tmp,0)
            del self.ind2[tmp]
            
            cnt = 1

        # while budget not reached
        while cnt < self.k:

            if len(self.ind2) == 0:                
                return self.C
            
            # D has a shape of (N,), N is the current number of instances in X2
            D = self.computeD()
            #print(len(D))
            
            # normalize D
            D = D/(sum(D,0)+np.array(sum(D,0)==0).astype(int))
            
            #cumulative sum of distances
            cumprobs = D.cumsum()

            
            if (self.m-cnt)==0:
                break
            elif len(cumprobs[cumprobs!=0]) == 0:
                # randomly select a sample
                rinds = random.sample(list(range(self.m-cnt)), self.k-cnt)

                # select the sample that is the most distant from the pool sample
                # rind = np.argmax(D)
                self.newind.extend(rinds)
                self.C.extend(self.X2[rinds])
                rinds.sort()
                for i in rinds[::-1]:
                    self.X2 = np.delete(self.X2,i,0)
                break
            else:
                r = random.random()
                if r == 1:
                    r = random.random()                   
                try:
                    rind = np.where(cumprobs >= r)[0][0]
                except:
                    print('D,', D, 'sum_D, ', sum(D))
                    print('cumprobs',cumprobs)
                    print(r)
                self.C.append(self.X2[rind])
                self.X2 = np.delete(self.X2,rind,0)
                self.newind.append(self.ind2[rind])
                del self.ind2[rind]

                cnt+=1
        return self.C

    def computeD(self):
        """

        :return: D a list of the distance of each sample to the reference points
        """
        
        D=[]
        for i in range(self.X2.shape[0]):
            tmpd = []
           # compute distance between this point and every point in self.C (reference points)
            for j in range(len(self.C)):
                d = np.sqrt(np.sum((self.X2[i]-self.C[j])**2/self.d))
                tmpd.append(d)                
            D.append(np.amin(tmpd))
        return D


# simple code to test the script, the csv file location should be defined properly if in use
if __name__ == "__main__":
    np.random.seed(0)
    dataset=pd.read_csv('Mall_Customers.csv')

    # data to select from
    data = dataset.iloc[:20, [3, 4]].values
    
    # if pool exists, otherwise pool = []
    pool = dataset.iloc[195:, [3, 4]].values

    budget = 50

    s = diversitySampling(data, pool = pool, budget = budget)
    s.updateCplus()
    # returns indices of # budget most diverse examples in data; if budget > size(data), returns only budget
    diversity = s.newind
    print(diversity)


