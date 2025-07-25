from __future__ import print_function
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import six
import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from tensorboardX import SummaryWriter
import data
import net

writer = SummaryWriter()
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--dimz', '-z', default=20, type=int,
                    help='dimention of encoded vector')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--test', action='store_true',
                    help='Use tiny datasets for quick tests')
args = parser.parse_args()
batchsize = args.batchsize
n_epoch = args.epoch
n_latent = args.dimz

writer.add_text('config', str(args))
print('GPU: {}'.format(args.gpu))
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)
if args.test:
    mnist['data'] = mnist['data'][0:100]
    mnist['target'] = mnist['target'][0:100]
    N = 30
else:
    N = 60000
x_train, x_test = np.split(mnist['data'], [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

model = net.VAE(784, n_latent, 500)
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy
optimizer = optimizers.Adam()
optimizer.setup(model)
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    perm = np.random.permutation(N)
    sum_loss = 0
    sum_rec_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        optimizer.update(model.get_loss_func(), x)
        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss,))
                o.write(g.dump())
            print('graph generated')
        writer.add_scalar('train/loss', model.loss, epoch * N + i)
        writer.add_scalar('train/rec_loss', model.rec_loss, epoch * N + i)
        sum_loss += float(model.loss.data) * len(x.data)
        sum_rec_loss += float(model.rec_loss.data) * len(x.data)
    print('train mean loss={}, mean reconstruction loss={}'
          .format(sum_loss / N, sum_rec_loss / N))
    sum_loss = 0
    sum_rec_loss = 0
    with chainer.no_backprop_mode():
        for i in six.moves.range(0, N_test, batchsize):
            x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
            loss_func = model.get_loss_func(k=10)
            loss_func(x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)
            writer.add_scalar('test/loss', model.loss, epoch * N_test + i)
            writer.add_scalar('test/rec_loss', model.rec_loss,
                              epoch * N_test + i)
            writer.add_image('reconstructed', model(
                x).reshape(-1, 1, 28, 28), epoch * N_test + i)
            writer.add_image('input', x.reshape(-1, 1, 28, 28),
                             epoch * N_test + i)
            del model.loss
    print('test  mean loss={}, mean reconstruction loss={}'
          .format(sum_loss / N_test, sum_rec_loss / N_test))

print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
model.to_cpu()

def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)

train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = chainer.Variable(np.asarray(x_train[train_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'train')
save_images(x1.data, 'train_reconstructed')

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = chainer.Variable(np.asarray(x_test[test_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'test')
save_images(x1.data, 'test_reconstructed')

z = chainer.Variable(np.random.normal(0, 1, (9, n_latent)).astype(np.float32))
x = model.decode(z)
save_images(x.data, 'sampled')