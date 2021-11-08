from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

import constants

def get_CXR_resnet_18_classifier():
    return ResNet(BasicBlock, num_classes=17)

def get_model(model_type):
    if model_type == 'classifier_resnet_18':
        return get_CXR_resnet_18_classifier()
    else:
        raise RuntimeError(f'Error: Specified model is not one of {constants.MODEL_TYPES}. Aborting.')