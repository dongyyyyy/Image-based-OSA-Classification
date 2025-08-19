from torchvision import models
import torch.nn as nn
from torch import Tensor
import torch

def _forward_impl_efficientNet_without_GAP(self, x: Tensor) -> Tensor:
    x = self.features(x)
    return x

def _forward_impl_ResNet_without_GAP(self, x: Tensor) -> Tensor:
    # print('x : ',x.get_device())
    # print('model : ',self.conv1.weight.device)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


def _forward_impl_vit_without_GAP(self, x: Tensor) -> Tensor:
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    # x = x[:, 0]

    # x = self.heads(x)

    return x

# swin transformer랑 convnext 두개 forward 함수 구현하기