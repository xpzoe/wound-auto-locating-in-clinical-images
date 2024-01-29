from torchvision import models
from torch import nn

def my_vgg16(load_weights):
    vgg16 = models.vgg16_bn(weights=load_weights)

    for param in vgg16.features.parameters():
        param.require_grad = True
    # for param in vgg16.features[24:].parameters():
    #    param.require_grad = True

    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer

    features.extend([nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12),
        )]) 
    
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    return vgg16

def my_vgg19(load_weights):
    vgg19 = models.vgg19_bn(weights=load_weights)

    for param in vgg19.features.parameters():
        param.require_grad = True
    #for param in vgg19.features[24:].parameters():
     #S   param.require_grad = True

    
    num_features = vgg19.classifier[-1].in_features
    features = list(vgg19.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12),
        )]) 
    vgg19.classifier = nn.Sequential(*features) # Replace the model classifier
    return vgg19

def my_resnet50(load_weights):
    resnet50 = models.resnet50(weights=load_weights)
  
    for param in resnet50.parameters():
        param.requires_grad = False
    
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12),
    )
    
    return resnet50

def my_inceptionv3(load_weights):
    inception_v3 = models.inception_v3(weights=load_weights)

    for param in inception_v3.parameters():
        param.requires_grad = False

    inception_v3.fc = nn.Sequential(
        nn.Linear(inception_v3.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12)
    )

    inception_v3.AuxLogits.fc = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12)
    )

    return inception_v3


def my_alexnet(load_weights):
    alexnet = models.alexnet(weights=load_weights)
    
    for param in alexnet.parameters():
        param.requires_grad = False
    
    num_fc = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Sequential(
        nn.Linear(num_fc, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 12),
    )
    
    return alexnet

