import torch
import torch.nn as nn



class CNN(nn.Module):
        
    def __init__(self, img_width, img_height, in_channels=1, num_conv=64, num_conv_blocks=1, dropout=0.0, num_classes=6):
        super(CNN, self).__init__()

        self.conv_sizes = [in_channels, *[num_conv*(2**i) for i in range(num_conv_blocks)]]
        self.conv_blocks = nn.Sequential(*[CNN._double_convolution(self.conv_sizes[i], self.conv_sizes[i+1]) for i in range(num_conv_blocks)])

        res_width = img_width
        for i in range(num_conv_blocks): res_width = res_width//2
        res_height = img_height
        for i in range(num_conv_blocks): res_height = res_height//2

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(in_features=res_width*res_height*self.conv_sizes[-1], out_features=num_classes)



    @staticmethod
    def _double_convolution(in_channels, out_channels):
        conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return conv_op
    

    def forward(self, x):

        x = self.conv_blocks(x) 

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.classifier(x)

        return x 
    

    def get_features(self, x):
        x = self.conv_blocks(x) 
        x = x.reshape(x.shape[0], -1)
        return x



class CNN_DualClassifier(nn.Module):
        
    def __init__(self, model1, model2, linear_size, dropout=0.0, num_classes=6):
        super(CNN_DualClassifier, self).__init__()

        self.model1 = model1
        self.model2 = model2

        for param in self.model1.parameters():
            param.requires_grad = True
        for param in self.model2.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(in_features=linear_size, out_features=num_classes)

    

    def forward(self, x1, x2):

        x1 = self.model1.get_features(x1) 
        x2 = self.model2.get_features(x2) 

        x = torch.cat([x1,x2], dim=1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x 
    





class CNN_2(nn.Module):
        
    def __init__(self, img_width, img_height, in_channels=1, num_conv=64, num_conv_blocks=1, dropout=0.0, linear_size=64, num_classes=6):
        super(CNN_2, self).__init__()

        self.conv_sizes = [in_channels, *[num_conv*(2**i) for i in range(num_conv_blocks)]]
        self.conv_blocks = nn.Sequential(*[CNN._double_convolution(self.conv_sizes[i], self.conv_sizes[i+1]) for i in range(num_conv_blocks)])

        res_width = img_width
        for i in range(num_conv_blocks): res_width = res_width//2
        res_height = img_height
        for i in range(num_conv_blocks): res_height = res_height//2

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(nn.Linear(in_features=res_width*res_height*self.conv_sizes[-1], out_features=linear_size),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=linear_size, out_features=num_classes)
                                        )


    

    def forward(self, x):

        x = self.conv_blocks(x) 

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.classifier(x)

        return x
    


class CNN_MultiChannel(nn.Module):
        
    def __init__(self, img_width, img_height, in_channels=4, num_conv=32, num_conv_blocks=1, dropout=0.0, num_classes=6):
        super(CNN_MultiChannel, self).__init__()

        self.conv_sizes = [1, *[num_conv*(2**i) for i in range(num_conv_blocks)]]
        self.conv_blocks = nn.Sequential(*[CNN._double_convolution(self.conv_sizes[i], self.conv_sizes[i+1]) for i in range(num_conv_blocks)])

        res_width = img_width
        for i in range(num_conv_blocks): res_width = res_width//2
        res_height = img_height
        for i in range(num_conv_blocks): res_height = res_height//2

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(in_features=in_channels*res_width*res_height*self.conv_sizes[-1], out_features=num_classes)

        self.softmax = nn.LogSoftmax(dim=1)
   

    def forward(self, x):

        parallel_conv = []

        for i in range(x.shape[1]):
            out = self.conv_blocks(x[:,i:i+1,:,:]) 
            out = out.reshape(out.shape[0], -1)
            parallel_conv.append(out)

        x = torch.cat(parallel_conv, dim=1)

        x = self.dropout(x)
        x = self.classifier(x)

        return self.softmax(x)
