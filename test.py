
import torch
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

batch_size = 2
num_channels = 3
image_size = 256
num_instance = 1
label_id=0
inputs = {
    'image': torch.rand((batch_size, num_channels, image_size, image_size)),
    'gt_bbox': torch.stack([torch.rand((num_instance, 4)) for _ in range(batch_size)]),
    'gt_class': torch.stack([torch.tensor(np.array(num_instance * [label_id])) for _ in range(batch_size)]),
    'gt_segm': torch.stack([torch.randint(0, 1, (num_instance, image_size, image_size), dtype=torch.float) for _ in range(batch_size)]),
}

# for k,v in inputs.items():
#   inputs[k] = v.cuda()


from main import get_model
rtdetr_model = get_model('./config.json')

# rtdetr_model.to('cuda')
x = rtdetr_model(inputs)

params = list(rtdetr_model.backbone.parameters()) + list(rtdetr_model.neck.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
optimizer.zero_grad()
print(x)

# x.mean().backward()
x['loss'].mean().backward()
