import torch
import segmentation_models_pytorch_new as smp

device = 'cuda'
model = smp.UnetCDdiffuse(encoder_name="resnet50", encoder_weights="imagenet",
                          in_channels=4, classes=1).to(device)
# concatenate t1 and t2
data = torch.ones((1, 8, 256, 256)).float().to(device)
p1, p2 = model(data)
print(p1.shape)
print(p2.shape)