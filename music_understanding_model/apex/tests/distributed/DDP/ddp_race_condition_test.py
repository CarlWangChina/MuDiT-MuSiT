import torch
import os
from torch.nn import Parameter
from torch.nn import Module
from apex.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser(description='allreduce hook example')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()

torch.set_printoptions(precision=10)
torch.manual_seed(args.local_rank)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = Parameter(torch.cuda.FloatTensor(4096*4096).fill_(1.0))
        self.b = Parameter(torch.cuda.FloatTensor(4096*4096).fill_(2.0))

    def forward(self, input):
        return (input*self.a)*self.b

model = Model()
model = DDP(model, delay_allreduce=True)
x = torch.cuda.FloatTensor(4096*4096)
passed = True

torch.cuda.cudart().cudaProfilerStart()
for i in range(10):
    x.fill_(i + args.local_rank)
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()
    print("i = {}".format(i))

    def info(name, param, val):
        expected = val*4096*4096*(2.*i+1)/2.
        actual = param.grad.data.sum().item()
        print(name+": grad.data_ptr() = {}, expected sum {}, got {}".format(
              param.grad.data_ptr(), expected, actual))
        return (expected == actual)

    if not info("model.a", model.module.a, 2.):
        passed = False
    if not info("model.b", model.module.b, 1.):
        passed = False

torch.cuda.cudart().cudaProfilerStop()
print("passed = ", passed)