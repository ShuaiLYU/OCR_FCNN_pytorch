import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import  datasets
import numpy as np
import pdb

vgg16_pretrained = models.vgg16(pretrained=True)

class VGG16_conv(torch.nn.Module):
    def __init__(self, n_classes=11, log_polar=False):
        super(VGG16_conv, self).__init__()
        self.input_size=[3,32,256]
        self.n_classes=n_classes #类别＋1（背景）
        self.log_polar=log_polar
        self.length_limit=16 #图片内字符长度的上限
        self.build_models()

    def build_models(self):
        # VGG16 (using return_indices=True on the MaxPool2d layers)
        #将输入高度和宽度都/8
        self.features = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv4
            # torch.nn.Conv2d(256, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(512, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(512, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # # conv5
            # torch.nn.Conv2d(512, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(512, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(512, 512, 3, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, stride=2, return_indices=True)
            )
        self.feature_outputs = [0] * len(self.features)
        self.pool_indices = dict()

        # 计算字符串长度网络
        self.count_character = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 32, 1024),  # 224x244 image pooled down to 7x7 from features
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(512, self.length_limit)
        )

        #识别字符串网络
        self.recognition_character=torch.nn.Sequential(
            torch.nn.Conv2d(256,1024,3,padding=1),
            torch.nn.ReLU(),
            #高度/2得到2，宽度减1得到2N-1 (要求输入高度为偶数，宽度为偶数)
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,1),return_indices=True),
            #高度/2得到1，宽度不变为2N-1，通道数为1024
            torch.nn.Conv2d(1024,1024, kernel_size=(2,1), stride=(1,1)),
            torch.nn.ReLU(),
            #（self.n_classes，1，2N-1）
            torch.nn.Conv2d(1024, self.n_classes, 3, padding=1),
            torch.nn.ReLU()
        )

        self._initialize_weights(self.log_polar)

    def dynamic_pooling(self,num_of_characters):
        width_input=self.input_size[2]/8
        width_output=num_of_characters*2
        filter_width=width_input+1-width_output
        filter_heigth=1
        self.resize_features=torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(filter_heigth,filter_width),stride=1)
        )


    def _initialize_weights(self, log_polar):
        # initializing weights using ImageNet-trained model from PyTorch
        if not log_polar:
            for i, layer in enumerate(vgg16_pretrained.features):
                if isinstance(layer, torch.nn.Conv2d) and i<17:
                    self.features[i].weight.data = layer.weight.data
                    self.features[i].bias.data = layer.bias.data
        else:
            checkpoint = torch.load('checkpoint.pth.tar')
            for i, layer in enumerate(vgg16_pretrained.features):
                if isinstance(layer, torch.nn.Conv2d) and i<17:
                    key_w = 'features.module.' + str(i) + '.weight'
                    key_b = 'features.module.' + str(i) + '.bias'
                    self.features[i].weight.data = checkpoint['state_dict'][key_w]
                    self.features[i].bias.data = checkpoint['state_dict'][key_b]

    def get_conv_layer_indices(self):
        return [0, 2, 5, 7, 10, 12, 14]

    def forward_features(self, x):
        output = x
        for i, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                self.feature_outputs[i] = output
                self.pool_indices[i] = indices
            else:
                output = layer(output)
                self.feature_outputs[i] = output
        return output
    def forward_recognition(self, x):
        output = x
        for i, layer in enumerate(self.recognition_character):

            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                #self.feature_outputs2[i] = output
                #self.pool_indices2[i] = indices
            else:
                output = layer(output)
                #self.feature_outputs2[i] = output

        return output



    def count_forward(self, x):
        output = self.forward_features(x)
        output = output.view(output.size()[0], -1)
        output = self.count_character(output)
        return output

    def forward(self,x,num_character=None):
        features = self.forward_features(x)
        if num_character is None:
            output_view = features.view(features.size()[0], -1)
            output = self.count_character(output_view)
            _, num_character=torch.max(output,1)
        self.dynamic_pooling(num_character)
        resizefeatures=self.resize_features(features)
        output=self.forward_recognition(resizefeatures)
        output.transpose_(0,3).squeeze_(3).squeeze_(2)
        #output= output.permute(1,0)
        return  num_character,output


    # def get_activation(self, x, layer_number, feature_map):
    #     output = x
    #     for i in range(layer_number + 1):
    #         if isinstance(layer, torch.nn.MaxPool2d):
    #             output, indices = layer(output)
    #         else:
    #             output = layer(output)
    #     return output[:, feature_map, :, :]
    #
    # def get_activation_l_n_f(self, x, l_n_f):
    #     activation_dict = {}
    #     output = x
    #     for i, layer in enumerate(self.features):
    #         if isinstance(self.features[i], torch.nn.MaxPool2d):
    #             output, indices = self.features[i](output)
    #         else:
    #             output = self.features[i](output)
    #             for j, k in enumerate(l_n_f):
    #                 if k[0] == i:
    #                     activation_dict[j] = output[:, k[1], :, :]
    #     return activation_dict





