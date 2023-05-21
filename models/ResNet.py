import torch
from torch import nn
from torch.nn import functional as F

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, padding=1, bias=False)

class Block(Module):
  def __init__(self, in_planes, planes):
    super(Block, self).__init__()
    self.in_planes = in_planes
    self.planes = planes

    self.conv1 = conv3x3(in_planes, planes)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv3x3(planes, planes)
    self.bn3 = nn.BatchNorm2d(planes)

    self.res_conv = nn.Sequential(
        conv1x1(in_planes, planes),
        nn.BatchNorm2d(planes),
    )

    self.relu = nn.LeakyReLU(0.1, inplace=True)
    self.pool = nn.MaxPool2d(2)

    self.meta_parameters = {}

  def forward(self, x, params=None, episode=None):
    out = self.conv1(x, params['conv1'])
    out = self.bn1(out, training=True) 
    out = self.relu(out)

    out = self.conv2(out, params['conv2'])
    out = self.bn2(out, training=True)
    out = self.relu(out)

    out = self.conv3(out, params['conv3'])
    out = self.bn3(out, training=True)

    x = self.res_conv(x, params['res_conv'], episode)

    out = self.pool(self.relu(out + x))
    return out

class ResNet12(nn.Module):
  def __init__(self, channels, bn_args):
    super(ResNet12, self).__init__()
    self.channels = channels

    self.meta_params = {}

    self.layer1 = Block(3, channels[0])
    self.layer2 = Block(channels[0], channels[1])
    self.layer3 = Block(channels[1], channels[2])
    self.layer4 = Block(channels[2], channels[3])
    
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.out_dim = channels[3]

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
          m.weight, mode='fan_out', nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)

  def get_out_dim(self):
    return self.out_dim

  def forward(self, x, params=None, episode=None):
    out = self.layer1(x, get_child_dict(params, 'layer1'), episode)
    out = self.layer2(out, get_child_dict(params, 'layer2'), episode)
    out = self.layer3(out, get_child_dict(params, 'layer3'), episode)
    out = self.layer4(out, get_child_dict(params, 'layer4'), episode)
    out = self.pool(out).flatten(1)
    return out
    