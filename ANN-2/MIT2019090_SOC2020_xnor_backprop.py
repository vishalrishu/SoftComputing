# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:29:17 2020

@author: VISHAL
"""
import random
import copy
import math
import matplotlib.pyplot as plt

class NN(object):
    def __init__(self, inputN, hiddenN, outputN):
        super(NN, self).__init__()
        self.i = inputN
        self.h = hiddenN
        self.o = outputN
    
        self.iToH = []
        for i in range(inputN):
            self.iToH.append([random.uniform(-1,1)] * hiddenN)
    
        self.hToO = []
        for i in range(hiddenN):
            self.hToO.append([random.uniform(-1,1)] * outputN)
    
        self.inputA = [1 for i in range(inputN)]
        self.hiddenA = [1 for i in range(hiddenN)]
        self.outputA = [1 for i in range(outputN)]

	
    def processInput(self,X):
        self.inputA = copy.deepcopy(X)
        for h in range(self.h):
            s = 0
            for i in range(self.i):
                s+= self.inputA[i] * self.iToH[i][h]
            self.hiddenA[h] = sigmoid(s)

        for o in range(self.o):
            s = 0
            for h in range(self.h):
                s+= self.hiddenA[h] * self.hToO[h][o]
            self.outputA[o] = sigmoid(s)

        return self.outputA

    def backprop(self,X,Y,out,N):
        o_delta = [0 for i in range(self.o)]
        for i in range(self.o):
            error = Y[i] - out[i]
            o_delta[i] = error * d_sigmoid(self.outputA[i])

        for i in range(self.h):
            for j in range(self.o):
                w_delta = self.hiddenA[i] * o_delta[j]
                self.hToO[i][j]+= N * w_delta 

        h_delta = [0 for i in range(self.h)]
        for j in range(self.h):
            error=0.0
            for k in range(self.o):
                error+=self.hToO[j][k] * o_delta[k]
            h_delta[j]= error * d_sigmoid(self.hiddenA[j])

        for i in range(self.i):
            for j in range(self.h):
                w_delta = h_delta[j] * self.inputA[i]
                self.iToH[i][j]+= N*w_delta

    def train(self,dataset,learningRate=0.3):
        error = 9999
        iteration = 0
        errors = []
        while error>0.01:
            error = 0
            for d in dataset:
                X = d[0]
                out = self.processInput(X)
                Y = d[1]
                error+=calculateError(out,Y)
                self.backprop(X,Y,out,learningRate)
                iteration+=1
                errors.append(error)
        print("Training ended at iteration= "+ str(iteration))
        self.test(dataset)
        plt.plot(range(len(errors)), errors)
        

    def test(self, patterns):
        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '-->', 'Expected', p[1] ,'--> Actual', self.processInput(inputs))
	
    def __str__(self):
        s = 'Input Layer Weights :' + str(self.iToH) + '\n'
        s += 'Hidden Layer Weights :' + str(self.hToO) + '\n'
		#s+= str(self.inputA) +'\n'+ str(self.hiddenA) +'\n' +str(self.outputA)
        return s

def calculateError(out,Y):
	error = 0
	for i in range(len(out)):
		error += math.pow(Y[i] - out[i],2)
	return error

def sigmoid(k):
	return 1/(1 + math.exp(-k))

def d_sigmoid(k):
	return k * (1-k)

if __name__ == '__main__':
    dataset = [[[0,0] , [1]], [[0,1] , [0]], [[1,0] , [0]], [[1,1] , [1]] ]
    nw = NN(2,2,1)
    nw.train(dataset,0.3)
    print(nw)
    