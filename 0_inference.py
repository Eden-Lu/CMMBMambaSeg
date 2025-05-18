

import torch 
from model.CMMBMambaSeg import CMMBMambaSeg

t1 = torch.rand(1, 4, 128, 128, 64).cuda()
# t2 = torch.rand(1, 4, 128, 128, 128).cuda()
# t3 = torch.rand(1, 4, 128, 128, 128).cuda()
# t4 = torch.rand(1, 4, 128, 128, 128).cuda()


model = CMMBMambaSeg(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).cuda()

# out = model(t1, t2, t3, t4)
out = model(t1)

print(out.shape)




