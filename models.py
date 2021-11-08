from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

import constants

def get_CXR_resnet_18_classifier():
    """
    See https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L301
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=17)

def get_model(model_type):
    if model_type == 'classifier_resnet_18':
        return get_CXR_resnet_18_classifier()
    else:
        raise RuntimeError(f'Error: Specified model is not one of {constants.MODEL_TYPES}. Aborting.')