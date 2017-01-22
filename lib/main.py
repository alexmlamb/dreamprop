#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")

import cPickle as pickle
import gzip
from loss import accuracy, crossent
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint

from load_cifar import CifarData


mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid

trainy = trainy.astype('int32')
validy = validy.astype('int32')

#config = {}
#config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"
#config['mb_size'] = 128
#config['image_width'] = 32

#cd_train = CifarData(config, segment="train")
#trainx = cd_train.images.reshape(50000,32*32*3) / 128.0 - 1.0
#trainy = cd_train.labels
#cd_valid = CifarData(config, segment="test")
#validx = cd_valid.images.reshape(10000,32*32*3) / 128.0 - 1.0
#validy = cd_valid.labels

print "train x", trainx

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print trainx.shape
print trainy.shape

print validx.shape
print validy.shape

num_steps = 10

def init_params_forward():

    p = {}

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024+784,1024)).astype('float32'))
    p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,10)).astype('float32'))
    p['Wo'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))


    return p

def init_params_synthmem():

    p = {}

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,784)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,10)).astype('float32'))

    return p


def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def bn(inp):
    return inp#(inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true):

    inp = join2(h, x_true)

    h1 = T.nnet.relu(T.dot(inp, p['W1']), alpha=0.02)
    h2 = T.nnet.relu(T.dot(h1, p['W2']), alpha=0.02)

    y_est = T.nnet.sigmoid(T.dot(h2, p['Wy']))

    h_next = T.dot(h2, p['Wo'])

    loss = crossent(y_est, y_true)

    return h_next, y_est, loss

def synthmem(p, h_next): 

    h1 = T.nnet.relu(T.dot(h_next, p['W1']), alpha=0.02)
    h2 = T.nnet.relu(T.dot(h1, p['W2']), alpha=0.02)

    h = T.dot(h2, p['Wh'])
    x = T.dot(h2, p['Wx'])
    y = T.nnet.sigmoid(T.dot(h2, p['Wy']))

    return h, x, y




params_forward = init_params_forward()
params_synthmem = init_params_synthmem()


'''
Set up the forward method and the synthmem_method
'''

x_true = T.matrix()
y_true = T.ivector()
h_in = T.matrix()

h_next, y_est, class_loss = forward(params_forward, h_in, x_true, y_true)

h_in_rec, x_rec, y_rec = synthmem(params_synthmem, h_next)

rec_loss = T.abs_(x_rec - x_true).mean() + T.abs_(y_est - y_rec).mean()

updates_forward = lasagne.updates.adam(rec_loss + class_loss, params_forward.values() + params_synthmem.values())

forward_method = theano.function(inputs = [x_true,y_true,h_in], outputs = [h_next, rec_loss, class_loss], updates=updates_forward)


#out_grad = T.grad(loss, h_in_init, known_grads = {h_out*1.0 : in_grad * T.gt(h_in_init,0.0)})


#param_grads = T.grad(net['loss'], params.values(), known_grads = {net['h_out']*1.0 : in_grad*1.0})





synthmem_method = theano.function(inputs = [])


for iteration in xrange(0,500000):
    r = randint(0,49900)
    x = trainx[r:r+64]
    y = trainy[r:r+64]

    h_in = np.zeros(shape=(64,m)).astype('float32')
    in_grad = np.zeros(shape=(64,m)).astype('float32')

    for j in range(num_steps):
        out = train_method(x,y,h_in,in_grad,j)
        h_in = out['h_out']
        in_grad = out['out_grad']
        if j == 0:
            in_grad *= 0.0
        if iteration % 100 == 0:
            print "grads", j, in_grad.mean()
            #print "prob", j, out['prob'][0].round(2)
            #print "h val", j, h_in.round(2)

    #using 500
    if iteration % 100 == 0:
        print "train acc", out['acc']
        print "train cost", out['loss']
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            for j in range(num_steps):
                valid_out = valid_method(validx[ind:ind+1000], validy[ind:ind+1000], h_in, j)
                h_in = valid_out['h_out']

            va.append(valid_out['acc'])
            vc.append(valid_out['loss'])

        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)



