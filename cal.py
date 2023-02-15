from models.GCoNet import GCoNet
import numpy as np
import time
import torch
from ptflops import get_model_complexity_info 

if __name__ == '__main__':
  net = GCoNet()
  net.cuda()
  model_parameters = filter(lambda p: p.requires_grad, net.parameters()) 
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(params)
  x = np.ones((1, 3, 224, 224), dtype=np.float32)
  x = torch.tensor(x, dtype=torch.float32, device='cuda')
  with torch.no_grad():
    s_t = time.time()
    _ = net(x)
    e_t = time.time()
    print('use time: {}'.format(e_t - s_t))
  # get flops
  with torch.no_grad():
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(macs, params)
