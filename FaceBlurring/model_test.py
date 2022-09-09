import torch
from models.mobilenet import FaceMobileNetV1, FaceMobileNetV2
from models.edgenext import EdgeNeXt, edgenext_xx_small
#from models.edgenext_bn_hs import edgenext_xx_small_bn_hs


##model = EdgeNext_xx_small
model = edgenext_xx_small()
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

torch.save(model, './edgenext_xx_small.pth')