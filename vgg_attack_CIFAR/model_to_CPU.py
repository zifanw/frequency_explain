'''
load torch model trained on GPU and save it using state_dict that can be loaded on CPU.
Must run on torch-GPU.
'''

import torch
import vgg

MODEL = "model_best.pth.tar"
PATH = "model_best.pth_CPU.tar"

# checkpoint = torch.load(MODEL, map_location='cpu')
checkpoint = torch.load(MODEL)
model = vgg.vgg19()
model.load_state_dict(checkpoint, strict=False)
# print(checkpoint['best_prec1'])
# print(checkpoint.keys())

torch.save(model.state_dict(), PATH)

