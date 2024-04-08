from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class EfficientnetWrapper(nn.Module):

    def __init__(self, append_channels_horizontally=True, linear_size=0, version=0, num_classes=6):
        super(EfficientnetWrapper, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained(f'efficientnet-b{version}')

        self.num_features = self.feature_extractor._fc.in_features

        self.feature_extractor._fc = nn.Identity()

        if linear_size == 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.num_features, out_features=num_classes),  
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.num_features, out_features=linear_size),  
                nn.BatchNorm1d(linear_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(linear_size, out_features=num_classes),  
                )
        
        self.dim_to_append_channels = -2 if append_channels_horizontally else -1

        self.freeze_feature_exctractor_weights()


    def freeze_feature_exctractor_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_exctractor_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    

    def forward(self, input):

        # append all image channels into single image
        layers = [input[:, i, :, :] for i in range(input.shape[1])]
        x = torch.cat(layers, dim=self.dim_to_append_channels)
        
        # add a channel dimension
        x=x.unsqueeze(1) 

        # make 3 channels for efficientnet
        x = torch.cat([x,x,x], dim=1)

        # print(x.shape)

        # apply model
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x
