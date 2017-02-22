#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")
import os

import time

os.system('/u/lambalex/.profile')
os.system('/u/lambalex/.bashrc')

import cPickle as pickle
import gzip
from loss import accuracy, crossent, expand, clip_updates
import theano
import theano.tensor as T
import numpy.random as rng
rng.seed(42)
import lasagne
import numpy as np
from random import randint
random.seed(42)

from nn_layers import param_init_lnlstm, lnlstm_layer, param_init_lngru, lngru_layer, param_init_fflayer, fflayer

from gan_objective import init_params_disc, disc

from utils import init_tparams


from viz import plot_images

from load_cifar import CifarData

srng = T.shared_randomstreams.RandomStreams(999)

def dropout(inp, p=0.5):
    return T.cast(srng.binomial(n=1,p=p,size=inp.shape),'float32') * inp

def cast(inp):
    return T.cast(inp, 'float32')

'''
1,100,1: go back 1 step
1,100,1,1,1: go back 3 steps
'''
def noneabove(rec_loss_lst,k,limit):
    for j in reversed(range(len(rec_loss_lst))):

        if rec_loss_lst[j] >= limit:
            return False

        if j <= k:
            return True


class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()



dataset = "mnist"
#dataset = "cifar"
#dataset = "ptb_char"

print "dataset", dataset

do_bptt = True

print "do bptt", do_bptt

num_steps = 783

m1 = 64
m2 = 128
m3 = 256

print "Number of units", m1, m2, m3

if dataset == "mnist":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid
    testx, testy = test

    print "train shapes", trainx.shape, trainy.shape
    print "valid shapes", validx.shape, validy.shape

    trainy = trainy.astype('int32')
    validy = validy.astype('int32')
    testy = testy.astype('int32')

    nf = 783/num_steps

    print "Number of pixels per step", nf

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

    param_init_lngru({}, params=p, prefix='gru1', nin=m1, dim=m1)

    param_init_lngru({}, params=p, prefix='gru2', nin=m1, dim=m2)

    param_init_lngru({}, params=p, prefix='gru3', nin=m2, dim=m3)

    tparams = init_tparams(p)

    #tparams['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,m,m)).astype('float32'))
    #tparams['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,m,m)).astype('float32'))
    #tparams['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(m,11)).astype('float32'))
    tparams['W0'] = theano.shared(0.03 * rng.normal(0,1,size=(1+nf,m1)).astype('float32'))

    #tparams['by'] = theano.shared(0.03 * rng.normal(0,1,size=(11,)).astype('float32'))
    tparams['b0'] = theano.shared(0.03 * rng.normal(0,1,size=(m1,)).astype('float32'))

    tparams['Wyout_1'] = theano.shared(0.03 * rng.normal(0,1,size=(m1+m2+m3,2048)).astype('float32'))
    tparams['Wyout_2'] = theano.shared(0.03 * rng.normal(0,1,size=(2048,11)).astype('float32'))
    
    tparams['byout_1'] = theano.shared(0.0 * rng.normal(0,1,size=(2048,)).astype('float32'))
    tparams['byout_2'] = theano.shared(0.0 * rng.normal(0,1,size=(11,)).astype('float32'))

    tparams['Wh_sm_1'] = theano.shared(0.03 * rng.normal(0,1,size = (m1,512)).astype('float32'))
    tparams['bh_sm_1'] = theano.shared(0.0 * rng.normal(0,1, size = (512,)).astype('float32'))

    tparams['Wh_sm_2'] = theano.shared(0.03 * rng.normal(0,1,size = (512,m1)).astype('float32'))
    tparams['bh_sm_2'] = theano.shared(0.0 * rng.normal(0,1, size = (m1,)).astype('float32'))


    tparams['Wh_sm_1_2'] = theano.shared(0.03 * rng.normal(0,1,size = (m2,512)).astype('float32'))
    tparams['bh_sm_1_2'] = theano.shared(0.0 * rng.normal(0,1, size = (512,)).astype('float32'))

    tparams['Wh_sm_2_2'] = theano.shared(0.03 * rng.normal(0,1,size = (512,m2)).astype('float32'))
    tparams['bh_sm_2_2'] = theano.shared(0.0 * rng.normal(0,1, size = (m2,)).astype('float32'))


    tparams['Wh_sm_1_3'] = theano.shared(0.03 * rng.normal(0,1,size = (m3,512)).astype('float32'))
    tparams['bh_sm_1_3'] = theano.shared(0.0 * rng.normal(0,1, size = (512,)).astype('float32'))

    tparams['Wh_sm_2_3'] = theano.shared(0.03 * rng.normal(0,1,size = (512,m3)).astype('float32'))
    tparams['bh_sm_2_3'] = theano.shared(0.0 * rng.normal(0,1, size = (m3,)).astype('float32'))


    return tparams


def join2(a,b):
    return T.concatenate([a,b], axis = 1)

def join3(a,b,c):
    return T.concatenate([a,b,c], axis = 1)

def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.01 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true, step):

    print "PROVIDING STEP"
    inp = join2(x_true,step*0.01)

    emb = T.dot(inp, p['W0']) + p['b0']

    h0 = lngru_layer(p,emb,{},prefix='gru1',mask=None,one_step=True,init_state=h,backwards=False)

    h_next = h0[0]

    return h_next

params_forward = init_params_forward()

##########################################################################
#BPTT forward method
##########################################################################

def dist(a,b):
    return T.sum(T.sqr(a-b))

bptt_lr = 0.0001
bptt_beta1 = 0.9

print "bptt params", bptt_lr, bptt_beta1


Xts = T.tensor3()
y = T.ivector()

h_initial = theano.shared(np.zeros((64,m1)).astype('float32'))
h_initial_2 = theano.shared(np.zeros((64,m2)).astype('float32'))
h_initial_3 = theano.shared(np.zeros((64,m3)).astype('float32'))
initial_step = np.zeros((64,1)).astype('float32')

params_forward['hi_1'] = h_initial
params_forward['hi_2'] = h_initial_2
params_forward['hi_3'] = h_initial_3

def one_step(xval,h_last,step):

    h_next = forward(params_forward, h_last, xval, y, step)

    return h_next, step + 1

[h_seq, step_seq], _ = theano.scan(fn=one_step, sequences=[Xts], outputs_info=[h_initial,initial_step])

steps_back = 0
print "PREDICTING steps back", steps_back

rec_loss = 0.0

####
#SYNTHMEM
####

print "BLOCKING GRADIENT"
print "USING SQR"

mult = 0.00001
print "mult", mult

print "50% drop in inp"
for k in range(50,700,50):
    h_pred_1 = T.nnet.relu(T.dot(dropout(h_seq[k + steps_back], 0.5), params_forward['Wh_sm_1']) + params_forward['bh_sm_1'])
    h_pred_2 = T.dot(h_pred_1, params_forward['Wh_sm_2']) + params_forward['bh_sm_2']
    rec_loss += mult * T.sum(T.sqr(consider_constant(h_seq[k]) - h_pred_2))

h_seq_2 = lngru_layer(params_forward,h_seq[::10],{},prefix='gru2',mask=None,one_step=False,init_state=h_initial_2,backwards=False)[0]

for k in range(0,50,10):
    h_pred_1 = T.nnet.relu(T.dot(dropout(h_seq_2[k + steps_back], 0.5), params_forward['Wh_sm_1_2']) + params_forward['bh_sm_1_2'])
    h_pred_2 = T.dot(h_pred_1, params_forward['Wh_sm_2_2']) + params_forward['bh_sm_2_2']
    rec_loss += mult * T.sum(T.sqr(consider_constant(h_seq_2[k]) - h_pred_2))

h_seq_3 = lngru_layer(params_forward,h_seq_2[::10],{},prefix='gru3',mask=None,one_step=False,init_state=h_initial_3,backwards=False)[0]

for k in range(0,7):
    h_pred_1 = T.nnet.relu(T.dot(dropout(h_seq_3[k + steps_back], 0.5), params_forward['Wh_sm_1_3']) + params_forward['bh_sm_1_3'])
    h_pred_2 = T.dot(h_pred_1, params_forward['Wh_sm_2_3']) + params_forward['bh_sm_2_3']
    rec_loss += mult * T.sum(T.sqr(consider_constant(h_seq_3[k]) - h_pred_2))

y_out_1 = T.nnet.relu(T.dot(join3(h_seq[-1], h_seq_2[-1], h_seq_3[-1]), params_forward["Wyout_1"]) + params_forward['byout_1'], alpha=0.02)
y_out = T.nnet.softmax(T.dot(y_out_1, params_forward["Wyout_2"]) + params_forward['byout_2'])

#Time x batch x feature
rl_grad = T.sum(T.sqr(T.grad(rec_loss, h_seq)), axis=(1,2))
cl_grad = T.sum(T.sqr(T.grad(crossent(y_out,y), h_seq)), axis=(1,2))

total_loss = crossent(y_out, y) + rec_loss

acc = accuracy(y_out, y)

print "USING rmsprop"
bptt_updates = lasagne.updates.rmsprop(total_loss, params_forward.values(), learning_rate=bptt_lr)

print "compile starting"
t0 = time.time()
bptt_train = theano.function([Xts, y], outputs = [acc, total_loss, rec_loss, rl_grad, cl_grad], updates=bptt_updates)
bptt_valid = theano.function([Xts, y], outputs = [acc, total_loss, rec_loss])
print "compile finished in", time.time() - t0

########################################################################
#Main loop section
########################################################################

h_forward_lst = [0]*num_steps
rec_loss_lst = [0]*num_steps

perm = rng.permutation(784)
permute = True
print "permute", permute

if permute:
    trainx = trainx.T[perm].T
    validx = validx.T[perm].T

print "RUNNING WITH FEWER EXAMPLES"

for iteration in xrange(0,200000):
    r = randint(0,100)

    x = trainx[r:r+64]
    y = trainy[r:r+64]

    #########
    #BPTT Method
    #########

    if do_bptt:

        t0 = time.time()
        xsteplst = []

        for j in range(0,num_steps):
            xsteplst.append(x[:,j*nf:(j+1)*nf])
            #if j == num_steps-1:
            #    ysteplst.append(y)
            #else:
            #    ysteplst.append(y*0+10)

        xts = np.asarray(xsteplst)
        #yts = np.asarray(ysteplst)


        last_acc, total_loss, total_rec_loss, rl_grad, cl_grad = bptt_train(xts,y)
        t1 = time.time()

    #using 500
    if iteration % 10 == 0:

        print "iteration", iteration
        print "time for bptt update", t1-t0
        print "rec loss", total_rec_loss
        print "Total Loss Train bptt", total_loss
        print "Acc Train bptt", last_acc

    if iteration % 1000 == 0 and iteration >= 500:

        print "rl grad", rl_grad.shape
        print rl_grad.tolist()
        print "cl grad", cl_grad.shape
        print cl_grad.tolist()

    if iteration % 50 == 0 and iteration >= 0:

        print "running validation"

        tv = time.time()

        va = []
        for ind in range(0,9900,64):

            x = validx[ind:ind+64]
            y = validy[ind:ind+64]

            xsteplst = []

            for j in range(0,num_steps):
                xsteplst.append(x[:,j*nf:(j+1)*nf])

            xts = np.asarray(xsteplst)

            acc, _, _ = bptt_valid(xts,y)

            va.append(acc)

        print "Iteration", iteration
        print "Valid time", time.time() - tv
        print "Valid accuracy", sum(va)/len(va)




