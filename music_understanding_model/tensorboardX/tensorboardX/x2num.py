from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import six
import numpy as np

def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        logging.warning('NaN or Inf found in input tensor.')
    return array

def make_np(x):
    if isinstance(x, list):
        return check_nan(np.array(x))
    if isinstance(x, np.ndarray):
        return check_nan(x)
    if isinstance(x, six.string_types):
        return check_nan(prepare_caffe2(x))
    if np.isscalar(x):
        return check_nan(np.array([x]))
    if 'torch' in str(type(x)):
        return check_nan(prepare_pytorch(x))
    if 'chainer' in str(type(x)):
        return check_nan(prepare_chainer(x))
    if 'mxnet' in str(type(x)):
        return check_nan(prepare_mxnet(x))
    raise NotImplementedError(
        'Got {}, but expected numpy array or torch tensor.'.format(type(x)))

def prepare_pytorch(x):
    import torch
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x

def prepare_theano(x):
    import theano
    pass

def prepare_caffe2(x):
    from caffe2.python import workspace
    x = workspace.FetchBlob(x)
    return x

def prepare_mxnet(x):
    x = x.asnumpy()
    return x

def prepare_chainer(x):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    return x