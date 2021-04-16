import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import random
from tqdm import tqdm
import torch.nn.functional as F

import  torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class MyEffientnet_b0(nn.Module):
    def __init__(self,model_name='efficientnet-b0',class_num=45,initfc_type='normal',gain=0.2):
        super(MyEffientnet_b0, self).__init__()
        model = EfficientNet.from_pretrained(model_name)

        self.model = model
        self.fc1 = nn.Linear(model._conv_head.out_channels, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

        if hasattr(self.fc1, 'bias') and self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias.data, 0.0)
        if initfc_type == 'normal':
            nn.init.normal_(self.fc1.weight.data, 0.0, gain)
        elif initfc_type == 'xavier':
            nn.init.xavier_normal_(self.fc1.weight.data, gain=gain)
        elif initfc_type == 'kaiming':
            nn.init.kaiming_normal_(self.fc1.weight.data, a=0, mode='fan_in')
        elif initfc_type == 'orthogonal':
            nn.init.orthogonal_(self.fc1.weight.data, gain=gain)


    def forward(self,x):
        x = self.model.extract_features(x)
        x = x * torch.sigmoid(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class MyResnet50(nn.Module):
    def __init__(self, ):
        super(MyResnet50, self).__init__()
        prev_resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(prev_resnet50.children())[:-1]))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.resnet50(x)
        out = out.squeeze(3)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

class MyResnet50_esemble(nn.Module):
    def __init__(self, ):
        super(MyResnet50_esemble, self).__init__()
        prev_resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(prev_resnet50.children())[:-1]))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.resnet50(x)
        out = out.squeeze(3)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MyEffientnet_b0_esemble(nn.Module):
    def __init__(self,model_name='efficientnet-b0',class_num=45,initfc_type='normal',gain=0.2):
        super(MyEffientnet_b0_esemble, self).__init__()
        model = EfficientNet.from_pretrained(model_name)

        self.model = model
        self.fc1 = nn.Linear(model._conv_head.out_channels, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

        if hasattr(self.fc1, 'bias') and self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias.data, 0.0)
        if initfc_type == 'normal':
            nn.init.normal_(self.fc1.weight.data, 0.0, gain)
        elif initfc_type == 'xavier':
            nn.init.xavier_normal_(self.fc1.weight.data, gain=gain)
        elif initfc_type == 'kaiming':
            nn.init.kaiming_normal_(self.fc1.weight.data, a=0, mode='fan_in')
        elif initfc_type == 'orthogonal':
            nn.init.orthogonal_(self.fc1.weight.data, gain=gain)


    def forward(self,x):
        x = self.model.extract_features(x)
        x = x * torch.sigmoid(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        super(MyEnsemble, self).__init__()
        'modelA -> resnet50'
        'modelB -> efficientb0'
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        self.classifier = nn.Linear(5, 1)

    def forward(self, x):
        x1 = self.modelA(x.clone())
        x2 = self.modelB(x.clone())
        x3 = self.modelC(x.clone())
        x4 = self.modelD(x.clone())
        x5 = self.modelE(x.clone())
        out = torch.cat((x1.unsqueeze(2), x2.unsqueeze(2), x3.unsqueeze(2), x4.unsqueeze(2), x5.unsqueeze(2)), dim = 2)
        out = self.classifier(out)
        out = out.squeeze(2)
        out = F.softmax(out, dim = 1)
        return out
