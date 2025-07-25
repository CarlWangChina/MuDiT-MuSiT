from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gc
import six
import time
from functools import wraps
from .writer import SummaryWriter
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.visdom_writer import VisdomWriter

vis_formats = {'tensorboard': SummaryWriter, 'visdom': VisdomWriter}

class TorchVis:
    def __init__(self, *args, **init_kwargs):
        self.subscribers = {}
        self.register(*args, **init_kwargs)

    def register(self, *args, **init_kwargs):
        formats = ['tensorboard'] if not args else args
        for format in formats:
            if self.subscribers.get(format) is None and format in vis_formats.keys():
                self.subscribers[format] = vis_formats[format](**init_kwargs.get(format, {}))

    def unregister(self, *args):
        for format in args:
            self.subscribers[format].close()
            del self.subscribers[format]
            gc.collect()

    def __getattr__(self, attr):
        for _, subscriber in six.iteritems(self.subscribers):
            def wrapper(*args, **kwargs):
                for _, subscriber in six.iteritems(self.subscribers):
                    if hasattr(subscriber, attr):
                        getattr(subscriber, attr)(*args, **kwargs)
            return wrapper
        raise AttributeError

    def __del__(self):
        for _, subscriber in six.iteritems(self.subscribers):
            subscriber.close()