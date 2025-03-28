import torch
import torch.nn as nn

# (optional) visualize the neural network
# from torchviz import make_dot

class vgg16(nn.Module):
    r"""same as torchvision.models.vgg16"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # conv4_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv4_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv5_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv5_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )
        
        self.feature_layers = {
            0: "conv1_1", 2: "conv1_2", 
            5: "conv2_1", 7: "conv2_2",
            10: "conv3_1", 12: "conv3_2", 14: "conv3_3",
            17: "conv4_1", 19: "conv4_2", 21: "conv4_3",
            24: "conv5_1", 26: "conv5_2", 28: "conv5_3",
        }

    def forward(self, x):
        x = self.features(x)       
        x = self.avgpool(x)        
        x = torch.flatten(x, 1)    
        x = self.classifier(x)     
        return x

    def collect(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.feature_layers.keys():
                features[self.feature_layers[index]] = x 
        return features
    
# (optional) visualize the neural network
# if __name__ == "__main__":
#     x = torch.randn(1, 3, 200, 200)
#     model = vgg16()
#     out = model(x)
#     graph = make_dot(out)
#     graph.render("network", view=False, format="jpg")