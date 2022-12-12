import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), # 2048
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Detector(nn.Module):
    def __init__(self, S = 7, B = 2):
        super(Detector, self).__init__()

        # Encoder.
        self.f = Model().f

        # Classifier.
        self.fc = nn.Linear(512, S * S * B * 5)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1) # TODO: why flatten?
        out = self.fc(feature)
        return out

if __name__ == '__main__':
    x = torch.randn((8, 3, 224, 224))
    print(x.shape)

    if False:
        model = Model()
        feature, output = model(x)
        print(x.shape, feature.shape, output.shape)
    else:
        pretrained_path = 'pretrained_model/trained_simclr_model_8.pth'
        model = Detector()
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        # freeze parameters
        for param in model.f.parameters():
            param.requires_grad = False
        # calculate output
        output = model(x)
        print('dimensions: ', x.shape, output.shape)


