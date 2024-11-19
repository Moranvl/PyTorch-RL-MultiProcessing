# import torch
# from torch import nn
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.linear = nn.Linear(10, 5)
#
#     def forward(self, x):
#         return self.linear(x)
#
# def copy_net_params(current_net, target_net):
#     for tar, cur in zip(target_net.parameters(), current_net.parameters()):
#         tar.data.copy_(cur.data)
#
# if __name__ == '__main__':
#     net1 = Net()
#     net2 = Net()
#     input_x = torch.randn(10)
#     print(net1(input_x))
#     print(net2(input_x))
#
#     net1.cuda()
#     # input_x.cuda()
#     copy_net_params(net1, net2)
#     # print(net1(input_x))
#     print(net2(input_x))

from multiprocessing import Queue, Process
import torch


def test_f(q):
    tensor = q.get()
    print(tensor)


if __name__ == '__main__':
    t = torch.Tensor([1234])
    q = Queue()
    p = Process(target=test_f, args=(q,))
    p.start()
    q.put(t)
    p.join()
