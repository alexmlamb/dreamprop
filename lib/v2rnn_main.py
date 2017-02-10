#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")
import os

os.system('/u/lambalex/.profile')
os.system('/u/lambalex/.bashrc')

import cPickle as pickle
import gzip
from loss import accuracy, crossent, expand
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint

from nn_layers import param_init_lnlstm, lnlstm_layer, param_init_lngru, lngru_layer, param_init_fflayer, fflayer

from utils import init_tparams


from viz import plot_images

from load_cifar import CifarData

'''
1,100,1: go back 1 step
1,100,1,1,1: go back 3 steps
'''
def noneabove(rec_loss_lst,k,limit):
    for j in reversed(range(len(rec_loss_lst))):

        if rec_loss_lst[j] >= limit:
            return False

        if j < k:
            return True

dataset = "mnist"
#dataset = "cifar"
#dataset = "ptb_char"

print "dataset", dataset

do_synthmem = True

print "do synthmem", do_synthmem

sign_trick = False

print "sign trick", sign_trick

use_class_loss_forward = 0.0

print "use class loss forward", use_class_loss_forward

only_y_last_step = True

print "only give y on last step", only_y_last_step

lr_f = 0.0001
beta1_f = 0.90

print "learning rate and beta forward_updates", lr_f, beta1_f

lr_s = 0.0001
beta1_s = 0.7

print "learning rate and beta synthmem_updates", lr_s, beta1_s

num_steps = 8

if dataset == "mnist":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid

    trainy = trainy.astype('int32')
    validy = validy.astype('int32')

    nf = 780/num_steps

elif dataset == "cifar":

    nf = 32*32*3

    config = {}
    config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"
    config['mb_size'] = 64
    config['image_width'] = 32

    cd_train = CifarData(config, segment="train")
    trainx = cd_train.images.reshape(50000,32*32*3) / 128.0 - 1.0
    trainy = cd_train.labels.astype('int32')
    cd_valid = CifarData(config, segment="test")
    validx = cd_valid.images.reshape(10000,32*32*3) / 128.0 - 1.0
    validy = cd_valid.labels.astype('int32')

    trainx = trainx.astype('float32')
    validx = validx.astype('float32')

print "train x", trainx

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print trainx.shape
print trainy.shape

print validx.shape
print validy.shape


print "doing deep bp"
print "using 1 layer forward net"
print "Number of steps", num_steps

def init_params_forward():

    p = {}

    param_init_lngru({}, params=p, prefix='gru1', nin=1024, dim=1024)

    tparams = init_tparams(p)

    tparams['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,1024)).astype('float32'))
    tparams['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,1024)).astype('float32'))
    tparams['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,11)).astype('float32'))
    tparams['W0'] = theano.shared(0.03 * rng.normal(0,1,size=(1024+nf,1024)).astype('float32'))

    return tparams

def init_params_synthmem():

    pa = {}

    param_init_lngru({}, params=pa, prefix='sm_gru1', nin=1024, dim=1024)
    
    p = init_tparams(pa)

    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    #p['Wh2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,1024)).astype('float32'))

    p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,2048)).astype('float32'))
    p['Wx2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,nf)).astype('float32'))

    p['Wy1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,1024)).astype('float32'))
    p['Wy2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,11)).astype('float32'))

    p['bh'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,)).astype('float32'))
    #p['bh2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,)).astype('float32'))
    
    p['bx'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,)).astype('float32'))
    p['bx2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,nf,)).astype('float32'))

    p['by1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,)).astype('float32'))

    return p


def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def ln(inp):
    return inp#(inp - T.mean(inp,axis=1,keepdims=True))# / (0.001 + T.std(inp,axis=1,keepdims=True))

def forward(p, h, x_true, y_true, i):

    i *= 0

    inp = join2(h, x_true)

    emb = T.dot(inp, p['W0'])

    h0 = lngru_layer(p,emb,{},prefix='gru1',mask=None,one_step=True,init_state=h[:,:1024],backwards=False)

    h1 = T.nnet.relu(ln(T.dot(h0[0], p['W1'][i])),alpha=0.02)
    h2 = T.nnet.relu(ln(T.dot(h1, p['W2'][i])),alpha=0.02)
    #h2 = h1

    y_est = T.nnet.softmax(T.dot(h2, p['Wy'][i]))

    #h_next = T.dot(h2, p['Wo'][i])
    h_next = h1

    loss = crossent(y_est, y_true)

    acc = accuracy(y_est, y_true)

    return h_next, y_est, loss, acc, y_est

def synthmem(p, h_next, i): 

    i *= 0
    
    emb = T.dot(h_next, p['Wh']) + p['bh']

    hn2 = lngru_layer(p,emb,{},prefix='sm_gru1',mask=None,one_step=True,init_state=h_next[:,:1024],backwards=False)
    #hn1 = T.tanh(T.dot(h_next, p['Wh'][i]) + p['bh'][i])
    #hn2 = T.tanh(T.dot(hn1[0], p['Wh2'][i]) + p['bh2'][i])

    xh1 = T.nnet.relu(ln(T.dot(h_next, p['Wx'][i]) + p['bx'][i]), alpha=0.02)
    x = T.dot(xh1, p['Wx2'][i]) + p['bx2'][i]
    
    yh1 = T.nnet.relu(ln(T.dot(h_next, p['Wy1'][i]) + p['by1'][i]), alpha=0.02)
    y = T.nnet.softmax(T.dot(yh1, p['Wy2'][i]))

    return hn2[0], x, y


params_forward = init_params_forward()
params_synthmem = init_params_synthmem()

'''
Set up the forward method and the synthmem_method
'''

x_true = T.matrix()
y_true = T.ivector()
h_in = T.matrix()
step = T.iscalar()

if only_y_last_step:
    y_true_use = T.switch(T.eq(step, num_steps-1), y_true, 10)
else:
    y_true_use = y_true


x_true_use = x_true
#x_true_use = T.switch(T.eq(step, 0), x_true, x_true*0.0)

h_next, y_est, class_loss,acc,probs = forward(params_forward, h_in, x_true_use, y_true_use,step)

h_in_rec, x_rec, y_rec = synthmem(params_synthmem, h_next,step)

print "0.1 mult"
rec_loss = 0.1 * (T.sqr(x_rec - x_true_use).sum() + T.sqr(h_in - h_in_rec).sum() + crossent(y_rec, y_true_use))

#should pull y_rec and y_true together!  

updates_forward = lasagne.updates.adam(rec_loss + use_class_loss_forward * class_loss, params_forward.values() + params_synthmem.values(),learning_rate=lr_f,beta1=beta1_f)

forward_method = theano.function(inputs = [x_true,y_true,h_in,step], outputs = [h_next, rec_loss, class_loss,acc,y_est], updates=updates_forward)
forward_method_noupdate = theano.function(inputs = [x_true,y_true,h_in,step], outputs = [h_next, rec_loss, class_loss,acc,probs])

'''
Goal: get a method that takes h[i+1] and dL/dh[i+1].  It runs synthmem on h[i+1] to get estimates of x[i], y[i], and h[i].  It then runs the forward on those values and gets that loss.  


'''

h_next = T.matrix()
g_next = T.matrix()

h_last, x_last, y_last = synthmem(params_synthmem, h_next,step)

x_last = x_last
y_last = y_last.argmax(axis=1)

h_next_rec, y_est, class_loss,acc,probs = forward(params_forward, h_last, x_last, y_last,step)

class_loss = class_loss * T.eq(step,num_steps-1)

if sign_trick:
    g_next_use = g_next * T.eq(T.sgn(h_next), T.sgn(h_next_rec))
else:
    g_next_use = g_next

hdiff = T.eq(T.sgn(h_next), T.sgn(h_next_rec)).mean()

h_rec_error = 0.1 * T.sum(T.sqr(h_next - h_next_rec))

g_last = T.grad(class_loss, h_last, known_grads = {h_next_rec*1.0 : g_next_use})
g_last_local = T.grad(class_loss, h_last)

param_grads = T.grad(class_loss * 1.0, params_forward.values(), known_grads = {h_next_rec*1.0 : g_next_use})

#Should we also update gradients through the synthmem module?
synthmem_updates = lasagne.updates.adam(param_grads, params_forward.values(),learning_rate=lr_s,beta1=beta1_s)

synthmem_method = theano.function(inputs = [h_next, g_next, step], outputs = [h_last, g_last, hdiff, g_last_local,x_last,y_last,h_rec_error], updates = synthmem_updates)

m = 1024

h_forward_lst = [0]*num_steps
rec_loss_lst = [0]*num_steps

for iteration in xrange(0,100000):
    r = randint(0,49900)

    x = trainx[r:r+64]
    y = trainy[r:r+64]

    h_in = np.zeros(shape=(64,m)).astype('float32')
    g_next = np.zeros(shape=(64,m)).astype('float32')

    h_final = h_forward_lst[-1]

    for (j,k) in zip(range(num_steps),reversed(range(num_steps))):
        x_step = x[:,j*nf:(j+1)*nf]
        h_next, rec_loss, class_loss,acc,y_est = forward_method(x_step,y,h_in,j)
        h_forward_lst[j] = h_next
        h_in = h_next
        rec_loss_lst[j] = rec_loss

        if iteration % 100 == 0:
            print "rec loss", j, rec_loss

        if do_synthmem and iteration > 1000 and noneabove(rec_loss_lst,k,100.0):
            if iteration % 100 == 1:
                print "step", k, rec_loss_lst[k]
            h_next_synthmem, g_last,hdiff,g_last_local,x_last_rec,y_last_rec,h_rec_error = synthmem_method(h_final,g_next,k)
            g_next = g_last
            h_final = h_next_synthmem

    #using 500
    if iteration % 100 == 0:
        print "limit of 100.0"

        print "========================================"
        print "train acc", acc
        print "train cost", class_loss
        print "train rec_loss", rec_loss
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            for j in range(num_steps):
                h_next,rec_loss,class_loss,acc,probs = forward_method_noupdate(validx[ind:ind+1000,j*nf:(j+1)*nf], validy[ind:ind+1000], h_in, j)
                h_in = h_next

            va.append(acc)
            vc.append(class_loss)

        print "REVERSED RANGE"
        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)


    if iteration % 500 == 0:
        print "testing on noisy input"
        h_in = np.zeros(shape=(1000,m)).astype('float32')
        x_val = rng.normal(size=(1000,nf)).astype('float32')
        for j in range(num_steps):
            h_next,rec_loss,class_loss,acc,probs = forward_method_noupdate(x_val, validy[ind:ind+1000], h_in, j)
            h_in = h_next

        print "acc noisy", acc
        print "probs", probs[0]


