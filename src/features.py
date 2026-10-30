"""Inputs of the scikit-learn models (M0 versions): raw pixels or hand-crafted colour / edge / texture features."""
import math

import numpy as np
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def rgb_to_hsv(x):
    """(N, 3, H, W) RGB in [0, 1] -> HSV in [0, 1]."""
    maximum, argmax = x.max(dim=1)
    minimum = x.min(dim=1).values
    delta = maximum - minimum
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    hue = torch.where(argmax == 0, ((g - b) / safe) % 6,
                      torch.where(argmax == 1, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    hue = torch.where(delta > 0, hue, torch.zeros_like(hue))
    saturation = torch.where(maximum > 0, delta / torch.where(maximum > 0, maximum, torch.ones_like(maximum)),
                             torch.zeros_like(maximum))
    return torch.stack([hue, saturation, maximum], dim=1)


def handcrafted_features(x, bins=(8, 4, 4), grid=4, orientations=8):
    """190 numbers per painting from (N, 3, H, W) pixels in [0, 1]:

    - HSV colour histogram (8 x 4 x 4 = 128): which colours, wherever they are (Pop Art's flat saturated colours,
      Baroque's dark browns, Ukiyo-e's pale backgrounds);
    - colour layout (grid x grid cells x RGB = 48): mean colour of each cell, e.g. "dark at the bottom";
    - edge orientations (8 orientations weighted by edge strength + edge density = 9): Cubism's straight
      facets, Impressionism's short scattered strokes;
    - texture (5): brightness mean and spread, edge-strength mean and spread, share of strong edges.
    """
    x = x.float().clamp(0, 1)
    hsv = rgb_to_hsv(x)
    q = [(hsv[:, c] * n).long().clamp(max=n - 1) for c, n in enumerate(bins)]
    codes = ((q[0] * bins[1] + q[1]) * bins[2] + q[2]).flatten(1)
    histogram = torch.stack([torch.bincount(c, minlength=math.prod(bins)) for c in codes]).float()
    histogram /= histogram.sum(1, keepdim=True)

    layout = F.adaptive_avg_pool2d(x, grid).flatten(1)

    gray = x.mean(1, keepdim=True)
    gx = F.conv2d(gray, SOBEL_X, padding=1)
    gy = F.conv2d(gray, SOBEL_X.transpose(2, 3), padding=1)
    magnitude = (gx ** 2 + gy ** 2).sqrt().flatten(1)
    angle = torch.atan2(gy, gx).flatten(1) % math.pi  # an edge and its opposite share an orientation
    bucket = (angle / math.pi * orientations).long().clamp(max=orientations - 1)
    edges = torch.zeros(len(x), orientations).scatter_add_(1, bucket, magnitude)
    edges = torch.cat([edges / (edges.sum(1, keepdim=True) + 1e-8), magnitude.mean(1, keepdim=True)], 1)

    strong = magnitude > magnitude.mean(1, keepdim=True) + magnitude.std(1, keepdim=True)
    texture = torch.stack([gray.flatten(1).mean(1), gray.flatten(1).std(1), magnitude.mean(1), magnitude.std(1),
                           strong.float().mean(1)], 1)
    return torch.cat([histogram, layout, edges, texture], 1).numpy()


def logistic_inputs(spec, x):
    """Rows given to a scikit-learn version for images x of shape (N, 3, H, W) in [0, 1]."""
    if spec.get("features", "pixels") == "handcrafted":
        return handcrafted_features(x)
    return np.asarray(x.reshape(len(x), -1).numpy())
