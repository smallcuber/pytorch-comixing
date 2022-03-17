import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim


def call_bn(bn, x):
    return bn(x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit


class ReshapedPreTrainedModel(nn.Module):
    def __init__(self, imported_model, n_outputs=10, dropout_rate=0.25, freeze_weights=False):
        # self.pre_trained_model = pre_trained_model
        super().__init__()
        self.pre_trained_model = imported_model
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate

        if freeze_weights:
            for param in self.pre_trained_model.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, self.n_outputs)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU()

    def forward(self, image):
        x = self.pre_trained_model(image)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


dict_models = {
    "CNN": CNN,
    "EfficientNet_v2_s": "tf_efficientnetv2_s_in21ft1k",
    "EfficientNet_v2_m": "tf_efficientnetv2_m_in21ft1k",
    "EfficientNet_v2_rw_t": "efficientnetv2_rw_t",
    "EfficientNet_b0": "efficientnet_b0",
    "EfficientNet_b1": "efficientnet_b1",
    "EfficientNet_b2": "efficientnet_b2",
    "EfficientNet_b3": "efficientnet_b3",
    "Densenet121": "densenet121",
    "Mobilenet_v2_035": "mobilenetv2_035",
    "Mobilenet_v2_100": "mobilenetv2_100",
}


def load_pretrained_model_by_name(model_name, is_pretrained=True):
    return torch.hub.load('rwightman/pytorch-image-models', dict_models[model_name], pretrained=is_pretrained)


def modify_pretrained_outputs(model, num_output=10, freeze_parameters=True):
    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False

    num_input = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_input, num_output, bias=True).to(device)
    # model.classifier.requires_grad = True
    return model
