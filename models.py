# from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision_resnet import ResNet, Bottleneck, BasicBlock
# import torchvision.models.resnet as tv_resnet
import torch.nn as nn
import torch

import constants

# class BasicBlock(nn.Module):
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = tv_resnet.conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = tv_resnet.conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         print('LOL \n\n\n')
#         print(out.shape)
#         print('GOTEM \n\n\n')
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

def get_CXR_resnet_18_classifier():
    """
    See https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L301
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=15)
    # --- First layer must take in a single channel ---
    # model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def get_model(model_type):
    if model_type == 'classifier_resnet_18':
        return get_CXR_resnet_18_classifier()
    else:
        raise RuntimeError(f'Error: Specified model is not one of {constants.MODEL_TYPES}. Aborting.')


def WeightedBCEWithLogitsLoss(weights=[1, 1]):
    """
    Weighted BCE loss which takes in raw logits.
    Not numerically stable...
    """
    def loss_fn(y, y_hat_logits, reduce_fn=torch.mean):
        y_hat = torch.sigmoid(y_hat_logits)
        # print(y)
        # print(y_hat)
        loss = -1 * (weights[0] * y * torch.log(y_hat) + weights[1] * (1 - y) * torch.log(1 - y_hat))
        # print(loss, loss.shape)
        loss = torch.sum(loss, dim=1)
        # print(loss, loss.shape)
        loss = reduce_fn(loss)
        # print(loss, loss.shape)
        return loss
        # return reduce_fn(
        #     torch.sum(
        #         -1 * (weights[0] * y * torch.log(y_hat) + weights[1] * (1 - y) * torch.log(1 - y_hat)),
        #     dim=1),
        # dim=0)
        
    return loss_fn