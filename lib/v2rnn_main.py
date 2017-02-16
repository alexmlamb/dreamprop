#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")
import os

os.system('/u/lambalex/.profile')
os.system('/u/lambalex/.bashrc')

import cPickle as pickle
import gzip
from loss import accuracy, crossent, expand, clip_updates
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint

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

do_synthmem = False

print "do synthmem", do_synthmem

do_forwardinc = False

print "do forwardinc", do_forwardinc

do_bptt = True

print "do bptt", do_bptt

sign_trick = False

print "sign trick", sign_trick

use_class_loss_forward = 1.0

print "use class loss forward", use_class_loss_forward

only_y_last_step = True

print "only give y on last step", only_y_last_step

lr_f = 0.0001
beta1_f = 0.9

print "learning rate and beta forward_updates", lr_f, beta1_f

lr_s = 0.0001
beta1_s = 0.9

print "learning rate and beta synthmem_updates", lr_s, beta1_s

num_steps = 783

limit = 200.0

print "BP rec limit", limit

m = 1024

print "Number of units", m

h_rec_weight = 0.1

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

    param_init_lngru({}, params=p, prefix='gru1', nin=m, dim=m)

    tparams = init_tparams(p)

    #tparams['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,m,m)).astype('float32'))
    #tparams['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,m,m)).astype('float32'))
    tparams['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(m,11)).astype('float32'))
    tparams['W0'] = theano.shared(0.03 * rng.normal(0,1,size=(m+nf,m)).astype('float32'))

    tparams['by'] = theano.shared(0.03 * rng.normal(0,1,size=(11,)).astype('float32'))
    tparams['b0'] = theano.shared(0.03 * rng.normal(0,1,size=(m,)).astype('float32'))


    return tparams

def init_params_synthmem():

    smu = 1024

    pa = {}

    param_init_lngru({}, params=pa, prefix='sm_gru1', nin=smu, dim=m)

    p = init_tparams(pa)

    print "number of synthmem units", smu

    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(m,smu)).astype('float32'))
    #p['Wh1'] = theano.shared(0.03 * rng.normal(0,1,size=(smu,smu)).astype('float32'))
    #p['Wh2'] = theano.shared(0.03 * rng.normal(0,1,size=(smu,m)).astype('float32'))

    #p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(m,m)).astype('float32'))
    p['Wx2'] = theano.shared(0.03 * rng.normal(0,1,size=(m,nf)).astype('float32'))

    #p['Wy1'] = theano.shared(0.03 * rng.normal(0,1,size=(m,m)).astype('float32'))
    p['Wy2'] = theano.shared(0.03 * rng.normal(0,1,size=(m,11)).astype('float32'))

    p['bh'] = theano.shared(0.0 * rng.normal(0,1,size=(smu,)).astype('float32'))
    #p['bh1'] = theano.shared(0.0 * rng.normal(0,1,size=(smu,)).astype('float32'))
    #p['bh2'] = theano.shared(0.0 * rng.normal(0,1,size=(m,)).astype('float32'))
    
    #p['bx'] = theano.shared(0.0 * rng.normal(0,1,size=(m,)).astype('float32'))
    p['bx2'] = theano.shared(0.0 * rng.normal(0,1,size=(nf,)).astype('float32'))

    #p['by1'] = theano.shared(0.0 * rng.normal(0,1,size=(m,)).astype('float32'))

    return p


def join2(a,b):
        return T.concatenate([a,b], axis = 1)

print "NOT USING LN"
def ln(inp):
    return inp# - T.mean(inp,axis=1,keepdims=True)) / (1.001 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true, i):

    i *= 0

    inp = join2(h, x_true)

    emb = T.dot(inp, p['W0']) + p['b0']

    h0 = lngru_layer(p,emb,{},prefix='gru1',mask=None,one_step=True,init_state=h[:,:m],backwards=False)

    #h1 = T.nnet.relu(ln(T.dot(h0[0], p['W1'][i])),alpha=0.02)
    #h2 = T.nnet.relu(ln(T.dot(h1, p['W2'][i])),alpha=0.02)
    #h2 = h1

    h2 = h0[0]

    y_est = T.nnet.softmax(T.dot(h2, p['Wy']) + p['by'])

    h_next = h0[0]

    loss = crossent(y_est, y_true)

    acc = accuracy(y_est, y_true)

    return h_next, y_est, loss, acc, y_est

def synthmem(p, h_next, i): 

    i *= 0

    dr = 0.1
    h_next = dropout(h_next,dr)

    print "USING DROPOUT IN SYNTHMEM", dr
    #print "THREE LAYER SYNTHMEM"

    emb = T.tanh(T.dot(h_next, p['Wh']) + p['bh'])
    #h1 = T.nnet.relu(T.dot(emb, p['Wh1']) + p['bh1'], alpha=0.02)
    #h2 = T.dot(h1, p['Wh2']) + p['bh2']


    #print "JUST USING 1 TANH LAYER SYNTHMEM"

    h2 = lngru_layer(p,emb,{},prefix='sm_gru1',mask=None,one_step=True,init_state=h_next[:,:m],backwards=False)[0]

    x = T.dot(h2, p['Wx2']) + p['bx2']
    
    y = T.nnet.softmax(T.dot(h2, p['Wy2']))

    return h2, x, y

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

hdiff = cast(T.eq(T.sgn(h_next), T.sgn(h_next_rec))).mean()

print "h rec weight", h_rec_weight

h_rec_error = np.asarray(h_rec_weight).astype('float32') * T.sum(T.sqr(h_next - h_next_rec)) + 0.0 * T.sum(cast(x_last)) + 0.0 * T.sum(cast(y_last))

#h_rec_error = theano.shared(np.asarray(0.0).astype('float32'))

g_last = T.grad(class_loss + h_rec_error, h_last, known_grads = {h_next_rec*1.0 : g_next_use})
g_last_local = T.grad(class_loss + h_rec_error, h_last)

param_grads = T.grad(class_loss * 1.0 + h_rec_error, params_forward.values() + params_synthmem.values(), known_grads = {h_next_rec*1.0 : g_next_use})

print "AM UPDATING SYNTHMEM PARAMS IN SYNTHMEM UPDATES"

#Should we also update gradients through the synthmem module?
synthmem_updates = lasagne.updates.adam(param_grads, params_forward.values() + params_synthmem.values(),learning_rate=lr_s,beta1=beta1_s)

synthmem_method = theano.function(inputs = [h_next, g_next, step], outputs = [h_last, g_last, hdiff, g_last_local,x_last,y_last,h_rec_error], updates = synthmem_updates)

##########################################################################
#BPTT forward method
##########################################################################

Xts = T.tensor3()
y = T.imatrix()
grad_sm_flag = T.iscalar()
params_disc = init_params_disc()

h_initial = theano.shared(np.zeros((64,m)).astype('float32'))
h_last = h_initial

bptt_lr = 0.001
bptt_beta1 = 0.9

print "bptt params", bptt_lr, bptt_beta1

hrw = 0.01

print "hrw", hrw

total_loss = 0.0
total_rec_loss = 0.0
disc_loss = 0.0

def dist(a,b):
    return T.sum(T.sqr(a-b))

for step in range(0, num_steps):
    h_next_rec, y_est, class_loss,acc,probs = forward(params_forward, h_last, Xts[step], y[step],step)

    h_next_rec_use = T.switch(grad_sm_flag, h_next_rec, consider_constant(h_next_rec))

    h_rec,x_rec,y_rec = synthmem(params_synthmem, h_next_rec_use, step)

    num_keep = 64
    print "ONLY RECONSTRUCTING H with gradient blocking", num_keep
    h_rec_loss = 1.0 * dist(consider_constant(h_last)[:,:num_keep], h_rec[:,:num_keep]) + 0.0 * dist(Xts[step], x_rec) + 0.0 * dist(expand(y[step],11), y_rec)

    #h_rec_loss = 1.0 * dist(disc(params_disc, consider_constant(h_last))[0], disc(params_disc, h_rec)[0]) + 0.0 * dist(Xts[step], x_rec) + 0.0 * dist(expand(y[step],11), y_rec)

    #disc_loss += disc(params_disc, h_last)[1].mean() - disc(params_disc, h_rec)[1].mean()

    h_last = h_next_rec
    last_acc = acc
    total_loss += class_loss + hrw * h_rec_loss
    total_rec_loss += h_rec_loss

#disc_updates = clip_updates(lasagne.updates.adam(disc_loss, params_disc.values(), learning_rate = 0.001, beta1 = 0.5), params_disc.values())

print "USING ADAM"
bptt_updates = lasagne.updates.adam(total_loss, params_forward.values() + params_synthmem.values(), learning_rate=bptt_lr)

print len(bptt_updates)

#bptt_updates.update(disc_updates)

#print "should be bigger", len(bptt_updates)

for v in bptt_updates.items():
    print v

#raise Exception("DONE")

#bptt_updates_clipped = clip_updates(bptt_updates, params_forward.values())

#print "turned on clipping"

t0 = time.time()
bptt_train = theano.function([Xts, y, grad_sm_flag], outputs = [last_acc,total_loss,total_rec_loss], updates=bptt_updates)
print time.time() - t0, "time to compile"

########################################################################
#Main loop section
########################################################################

h_forward_lst = [0]*num_steps
rec_loss_lst = [0]*num_steps

rng.seed(42)
perm = rng.permutation(784)
permute = True
print "permute", permute

if permute:
    trainx = trainx.T[perm].T
    validx = validx.T[perm].T

for iteration in xrange(0,200000):
    r = randint(0,49900)

    x = trainx[r:r+64]
    y = trainy[r:r+64]


    h_in = np.zeros(shape=(64,m)).astype('float32')
    g_next = np.zeros(shape=(64,m)).astype('float32')

    h_final = h_forward_lst[-1]

    for (j,k) in zip(range(num_steps),reversed(range(num_steps))):

        if do_forwardinc:

            x_step = x[:,j*nf:(j+1)*nf]
            h_next, rec_loss, class_loss,acc,y_est = forward_method(x_step,y,h_in,j)
            h_forward_lst[j] = h_next
            h_in = h_next
            rec_loss_lst[j] = rec_loss

            if iteration % 100 == 0:
                print "rec loss", j, rec_loss

        if do_synthmem and iteration > 1000 and noneabove(rec_loss_lst,k,limit):
            if iteration % 100 == 1:
                print "step", k, rec_loss_lst[k]
            h_next_synthmem, g_last,hdiff,g_last_local,x_last_rec,y_last_rec,h_rec_error = synthmem_method(h_final,g_next,k)
            g_next = g_last
            h_final = h_next_synthmem

    #########
    #BPTT Method
    #########

    if do_bptt:

        xsteplst = []
        ysteplst = []

        for j in range(0,num_steps):
            xsteplst.append(x[:,j*nf:(j+1)*nf])
            if j == num_steps-1:
                ysteplst.append(y)
            else:
                ysteplst.append(y*0+10)

        xts = np.asarray(xsteplst)
        yts = np.asarray(ysteplst)

        wn = 0.0
        for p in params_synthmem.values():
            wn += (p.get_value()**2).sum()

        wnf = 0.0
        for p in params_forward.values():
            wnf += (p.get_value()**2).sum()

        if iteration > 0:
            grad_sm_flag = 1
        else:
            grad_sm_flag = 0

        last_acc, total_loss, total_rec_loss = bptt_train(xts,yts, grad_sm_flag)



    #using 500
    if iteration % 100 == 0:

        print "WEIGHT NORM", wn
        print "WEIGHT NORM FORWARD", wnf
        print "rec loss", total_rec_loss
        print "Total Loss Train bptt", total_loss
        print "Acc Train bptt", last_acc
        print "========================================"

        if do_forwardinc:
            print "train acc", acc
            print "train cost", class_loss
            print "train rec_loss", rec_loss
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            
            for j in range(num_steps):
                vx = validx[ind:ind+1000,j*nf:(j+1)*nf]
                h_next,rec_loss,class_loss,acc,probs = forward_method_noupdate(vx, validy[ind:ind+1000], h_in, j)
                h_in = h_next

            va.append(acc)
            vc.append(class_loss)

        print "REVERSED RANGE"
        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)




