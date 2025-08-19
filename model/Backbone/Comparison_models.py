from torchvision import models
import torch.nn as nn
from torch import Tensor
import torch
from collections import OrderedDict

def models_comparison_single_view(model_architecture='efficient_b0',num_classes=2,pretrained=False,dropout=0.2,image_size=448):
    if pretrained:
        if model_architecture == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            # model = resnet18(class_num=args.num_classes)
        elif model_architecture == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1')
            # model = resnet34(class_num=args.num_classes)
        elif model_architecture == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
        elif model_architecture == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1')
        elif model_architecture == 'efficient_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        elif model_architecture == 'efficient_b1':
            model = models.efficientnet_b1(weights='IMAGENET1K_V1')
        elif model_architecture == 'efficient_b2':
            model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        elif model_architecture == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1',image_size=image_size)
        elif model_architecture == 'vit_b_16_384':
            model = models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1',image_size=image_size)
        elif model_architecture == 'swin_t':
            model = models.swin_t(weights='IMAGENET1K_V1')
        elif model_architecture == 'swin_s':
            model = models.swin_s(weights='IMAGENET1K_V1')
        elif model_architecture == 'swin_b':
            model = models.swin_b(weights='IMAGENET1K_V1')
        elif model_architecture == 'convnext_b':
            model = models.convnext_base(weights='IMAGENET1K_V1')
        if model_architecture == 'efficient_b0' or model_architecture == 'efficient_b1' or model_architecture == 'efficient_b2':
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )
        if model_architecture == 'resnet18' or model_architecture == 'resnet34' or model_architecture == 'resnet50' or model_architecture == 'resnet101':
            model.fc = nn.Linear(model.fc.in_features,num_classes)
        if model_architecture == 'vit_b_16' or model_architecture == 'vit_b_16_384':
            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            heads_layers["head"] = nn.Linear(model.hidden_dim, num_classes)
            model.heads = nn.Sequential(heads_layers)
        if model_architecture.split('_')[0] == 'swin':
            model.head = nn.Linear(model.head.in_features, num_classes)
        if model_architecture.split('_')[0] == 'convnext':
            # print(f'model.classifier[-1].in_features = {model.classifier[-1].in_features}')
            model.classifier = nn.Sequential(
                nn.LayerNorm([model.classifier[-1].in_features,1,1],eps=1e-6), 
                nn.Flatten(1), 
                nn.Linear(model.classifier[-1].in_features, num_classes)
                )

    else:
        if model_architecture == 'efficient_b0':
            model = models.efficientnet_b0(weights=None,num_classes=2)
        elif model_architecture == 'efficient_b1':
            model = models.efficientnet_b1(weights=None,num_classes=2)
        elif model_architecture == 'efficient_b2':
            model = models.efficientnet_b2(weights=None,num_classes=2)
        elif model_architecture == 'resnet18':
            model = models.resnet18(weights=None,num_classes=2)
            # model = resnet18(class_num=args.num_classes)
        elif model_architecture == 'resnet34':
            model = models.resnet34(weights=None,num_classes=2)
            # model = resnet34(class_num=args.num_classes)
        elif model_architecture == 'resnet50':
            model = models.resnet50(weights=None,num_classes=2) 
        elif model_architecture == 'resnet101':
            model = models.resnet101(weights=None,num_classes=2)
        elif model_architecture == 'vit_b_16':
            model = models.vit_b_16(weights=None,num_classes=2)
        elif model_architecture == 'swin_t':
            model = models.swin_t(weights=None,num_classes=2)
        elif model_architecture == 'swin_s':
            model = models.swin_s(weights=None,num_classes=2)
        elif model_architecture == 'swin_b':
            model = models.swin_b(weights=None,num_classes=2)
        elif model_architecture == 'convnext_b':
            model = models.convnext_base(weights=None,num_classes=2)
    return model