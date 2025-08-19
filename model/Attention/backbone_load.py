from . import *
from utils.Re_define_forward import _forward_impl_efficientNet_without_GAP,_forward_impl_ResNet_without_GAP
from model.Backbone.EfficientNet import efficientnet_b2_without_GAP,efficientnet_b0_without_GAP

def backbone_without_GAP(model_architecture='efficientNet_b0',weight_files=None,pretrained=True,num_classes=2,dropout=0.2):
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
            model = efficientnet_b0_without_GAP(weights='IMAGENET1K_V1')
        elif model_architecture == 'efficient_b1':
            model = models.efficientnet_b1(weights='IMAGENET1K_V1')
        elif model_architecture == 'efficient_b2':
            # print('BBB')
            model = efficientnet_b2_without_GAP(weights='IMAGENET1K_V1')
        elif model_architecture == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1')
        elif model_architecture == 'swin_t':
            model = models.swin_t(weights='IMAGENET1K_V1')
        elif model_architecture == 'swin_s':
            model = models.swin_s(weights='IMAGENET1K_V1')
        elif model_architecture == 'swin_b':
            model = models.swin_b(weights='IMAGENET1K_V1')
        elif model_architecture == 'convnext_b':
            model = models.convnext_base(weights='IMAGENET1K_V1')
        if model_architecture.split('_')[0] == 'efficient':
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )
        if model_architecture == 'resnet18' or model_architecture == 'resnet34' or model_architecture == 'resnet50' or model_architecture == 'resnet101':
            model.fc = nn.Linear(model.fc.in_features,num_classes)
        if model_architecture == 'vit_b_16':
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
            model = efficientnet_b0_without_GAP(weights=None,num_classes=2)
        elif model_architecture == 'efficient_b1':
            model = models.efficientnet_b1(weights=None,num_classes=2)
        elif model_architecture == 'efficient_b2':
            print('AAA')
            model = efficientnet_b2_without_GAP(weights=None,num_classes=2)
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
    
    # checking dimension of final layer 
    if model_architecture.split('_')[0] == 'efficient':
        output_dim = model.classifier[1].in_features
    if model_architecture == 'resnet18' or model_architecture == 'resnet34' or model_architecture == 'resnet50' or model_architecture == 'resnet101':
        output_dim = model.fc.in_features
    if model_architecture == 'vit_b_16':
        output_dim = model.hidden_dim
    if model_architecture.split('_')[0] == 'swin':
        output_dim = model.head.in_features
    if model_architecture.split('_')[0] == 'convnext':
        # print(f'model.classifier[-1].in_features = {model.classifier[-1].in_features}')
        output_dim = model.classifier[-1].in_features
    if weight_files is None:
        # print('Here!! AA')
        return model, output_dim
    else:
        # print('Here!! BB')
        check = torch.load(weight_files,weights_only=True)
        state_dict = check['model_state_dict']
        
        model.load_state_dict(state_dict, strict = True)

    return model, output_dim


      
          
    # if model_architecture == 'efficient_b0' or model_architecture == 'efficient_b1' or model_architecture == 'efficient_b2':
    #     model.forward = _forward_impl_efficientNet_without_GAP.__get__(model, models.EfficientNet)
    #     # remove Global Average pooling and classifier layer
    #     del model.avgpool
    #     del model.classifier
    #     ###############################################
    # if model_architecture == 'resnet18' or model_architecture == 'resnet34' or model_architecture == 'resnet50' or model_architecture == 'resnet101':
    #     model.forward = _forward_impl_ResNet_without_GAP.__get__(model, models.ResNet)
    #     # remove Global Average pooling and classifier layer
    #     del model.avgpool
    #     del model.fc
    #     ###############################################
    