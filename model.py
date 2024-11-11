import torch
from torch import nn
from torch.nn import Module
from torch import Tensor
import inspect
import logging
import importlib
import torch.nn.functional as F
logger = logging.getLogger(__name__)


# Taken from FedFisher for CIFAR10
class FedNet(nn.Module):
    def __init__(self, in_channels, bias=False, num_classes=10):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(64 * 5 * 5, 512, bias=bias)
        self.fc2 = nn.Linear(512, 128, bias=bias)
        self.fc3 = nn.Linear(128, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RFFL_CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(RFFL_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# for MNIST 32*32
class MNIST_Net(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_Net(nn.Module):

	def __init__(self, device=None):
		super(MLP_Net, self).__init__()
		self.fc1 = nn.Linear(1024, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(-1,  1024)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class TwoCNN(torch.nn.Module): # McMahan et al., 2016; 1,663,370 parameters
    """"""
    def __init__(self, in_channels, num_classes):
        super(TwoCNN, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.Conv2d(in_channels=256, out_channels=256 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=(256 * 2) * (7 * 7), out_features=512, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class TwoCNNv2(torch.nn.Module): # McMahan et al., 2016; 1,663,370 parameters
    """"""
    def __init__(self, in_channels, num_classes):
        super(TwoCNNv2, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(5, 5), padding=2, stride=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = torch.nn.Linear(in_features=(256 * 2) * 7 * 7, out_features=512, bias=True)

        self.fc2 = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(-1, 256 * 2 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##########
# ResNet #
##########
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(planes),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(planes),
        )

        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        x = self.features(x) + self.shortcut(x)
        x = torch.nn.functional.relu(x)
        return x


__all__ = ["ResNet10", "ResNet18", "ResNet34"]

CONFIGS = {"ResNet10": [1, 1, 1, 1], "ResNet18": [2, 2, 2, 2], "ResNet34": [3, 4, 6, 3]}


class ResNet(torch.nn.Module):
    def __init__(self, config, block, in_channels, hidden_size, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            self._make_layers(block, 64, config[0], stride=1),
            self._make_layers(block, 128, config[1], stride=2),
            self._make_layers(block, 256, config[2], stride=2),
            self._make_layers(block, 512, config[3], stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear((7 * 7) * 512, self.num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.hidden_size, planes, stride))
            self.hidden_size = planes
        return torch.nn.Sequential(*layers)


class ResNet10(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet10, self).__init__(
            CONFIGS["ResNet10"], ResidualBlock, in_channels, hidden_size, num_classes
        )


class ResNet18(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet18, self).__init__(
            CONFIGS["ResNet18"], ResidualBlock, in_channels, hidden_size, num_classes
        )


class ResNet34(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet34, self).__init__(
            CONFIGS["ResNet34"], ResidualBlock, in_channels, hidden_size, num_classes
        )


MODEL_MAP = {
    "twocnn": TwoCNN,
    "twocnnv2": TwoCNNv2,
    "fednet": FedNet,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "rffl_cnn": RFFL_CNN,
}


#########################
# Weight initialization #
#########################
def init_weights(model: Module, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """

    def init_func(m: Module):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight"):
                if isinstance(m.weight, Tensor):
                    torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, "bias"):
                if isinstance(m.bias, Tensor):
                    torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Conv") != -1 or classname.find("Linear") != -1:
            if hasattr(m, "weight"):
                if isinstance(m.weight, Tensor):
                    if init_type == "normal":
                        torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
                    elif init_type == "xavier":
                        torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                    elif init_type == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                    elif init_type == "kaiming":
                        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                    elif init_type == "orthogonal":
                        torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                    elif init_type == "none":  # uses pytorch's default init method
                        m.reset_parameters()  # type: ignore
                    else:
                        raise NotImplementedError(
                            f"[ERROR] Initialization method {init_type} is not implemented!"
                        )
            if hasattr(m, "bias") and m.bias is not None:
                if isinstance(m.bias, Tensor):
                    torch.nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_model(
    model_name, in_channels: int, num_classes: int, init_type: str, init_gain=-1.0
) -> Module:
    # initialize the model class
    model = MODEL_MAP[model_name](in_channels=in_channels, num_classes=num_classes)

    init_weights(model, init_type, init_gain)

    logger.info(
        f"[MODEL] Initialized model: {model_name}; (Initialization type: {init_type.upper()})"
    )
    return model
