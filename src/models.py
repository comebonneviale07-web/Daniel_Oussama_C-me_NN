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
        layers.append(nn.Linear(incoming, outgoing))
        if spec.get("batch_norm"):
            layers.append(nn.BatchNorm1d(outgoing))
        layers.append(nn.Sigmoid() if spec["activation"] == "sigmoid" else nn.ReLU())
        if spec["dropout"]:
            layers.append(nn.Dropout(spec["dropout"]))
    layers.append(nn.Linear(widths[-1], classes))
    return nn.Sequential(*layers)


def preprocessing(name, normalization=None, train=False):
    """Deterministic preprocessing; with train=True, the version's "augmentation" is added (training images only).

    "flip": left-right mirror (a mirrored painting keeps its style).
    "paintings": what a visitor's camera changes: framing, mirror, a slightly tilted phone, an oblique view,
    gallery lighting. No vertical flip or large rotation (a painting is never upside down) and no hue change
    (colour is a cue of the style).
    """
    spec = get_version(name)
    augmentation = spec.get("augmentation") if train else None
    if augmentation not in (None, "flip" if spec["normalization"] != "imagenet" else "paintings"):
        raise ValueError(f"{name}: augmentation {augmentation!r} is not available for this kind of model")
    if spec["normalization"] == "imagenet":
        if augmentation == "paintings":
            weights = models.ResNet18_Weights.DEFAULT.transforms()
            return transforms.Compose([
                transforms.RandomResizedCrop(spec["input_size"], scale=(.6, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomPerspective(distortion_scale=.2, p=.3),
                transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(weights.mean, weights.std)])
        return models.ResNet18_Weights.DEFAULT.transforms()
    steps = [transforms.Resize((spec["input_size"], spec["input_size"]))]
    if augmentation == "flip":
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    if spec["normalization"] == "training_channel":
        if normalization is None:
            raise ValueError(f"{name} requires the training normalization stored in its checkpoint")
        steps.append(transforms.Normalize(normalization["mean"], normalization["std"]))
    return transforms.Compose(steps)
