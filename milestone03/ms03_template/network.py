from torch import nn
import timm

class CustomCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def get_network(arch_name, num_channels, num_classes, pretrained=False):
    """
    Returns the specified network with adjustments for multispectral images and 
    classification tasks.
    
    Args:
        arch_name (str): Name of the architecture ('ResNet18', 'CustomCNN', 'ConvNeXt-Nano', 'ViT-Tiny').
        num_channels (int): Number of input channels (e.g., 3 for RGB, >3 for multispectral images).
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained model for supported architectures.
    
    Returns:
        nn.Module: Initialized neural network.
    """
    # ResNet18
    if arch_name == "ResNet18":
        model = timm.create_model("resnet18", pretrained=pretrained)
        # Adjust input to handle multispectral channels
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust output layer to the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ConvNeXt-Nano
    elif arch_name == "ConvNeXt-Nano":
        model = timm.create_model("convnext_nano", pretrained=pretrained)
        # Adjust input to handle multispectral channels
        model.stem[0] = nn.Conv2d(num_channels, model.stem[0].out_channels, kernel_size=4, stride=4, padding=0)
        # Adjust output layer to the number of classes
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    # ViT-Tiny
    elif arch_name == "ViT-Tiny":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)
        # Adjust patch embedding layer to use patch size of 8 and handle multispectral channels
        model.patch_embed = timm.models.vision_transformer.PatchEmbed(
            img_size=112, patch_size=8, in_chans=num_channels, embed_dim=model.embed_dim
        )
        # Adjust output layer to the number of classes
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    # CustomCNN
    elif arch_name == "CustomCNN":
        return CustomCNN(num_channels, num_classes)

    # If the architecture name is not recognized
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")