import numpy as pb


def DepthMap(bl, f, img):
    depth = (bl * f) / (img)
    return depth
