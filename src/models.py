"""Shared builders; each version's architecture and preprocessing live in its config.json."""
from torch import nn
from torchvision import models, transforms

from .versions import get_version


def build_model(name, classes, pretrained=True):
    spec = get_version(name)
    if spec["kind"] == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        if spec["frozen"]:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if spec["kind"] != "mlp":
        raise ValueError(f"{name} is trained with scikit-learn, not a torch network")
    widths = [3 * spec["input_size"] ** 2, *spec["hidden_sizes"]]
    layers = [nn.Flatten()]
    for incoming, outgoing in zip(widths, widths[1:]):
        layers.extend([nn.Linear(incoming, outgoing), nn.Sigmoid() if spec["activation"] == "sigmoid" else nn.ReLU()])
        if spec["dropout"]:
            layers.append(nn.Dropout(spec["dropout"]))
    layers.append(nn.Linear(widths[-1], classes))
    return nn.Sequential(*layers)


def preprocessing(name, normalization=None):
    spec = get_version(name)
    if spec["normalization"] == "imagenet":
        return models.ResNet18_Weights.DEFAULT.transforms()
    steps = [transforms.Resize((spec["input_size"], spec["input_size"])), transforms.ToTensor()]
    if spec["normalization"] == "training_channel":
        if normalization is None:
            raise ValueError(f"{name} requires the training normalization stored in its checkpoint")
        steps.append(transforms.Normalize(normalization["mean"], normalization["std"]))
    return transforms.Compose(steps)
