import numpy as pb
import cv2


def Search(column, window, width):
    range = 75
    padding = window // 2
    right_bound = column
    left_bound = column - range
    if left_bound < padding:
        left_bound = padding
    step = 1
    return left_bound, right_bound, step


def Pad(ipbut, padding):
    rows = ipbut.shape[0]
    columns = ipbut.shape[1]
    output = pb.zeros((rows + padding * 2, columns + padding * 2), dtype=float)
    output[padding: rows + padding, padding: columns + padding] = ipbut
    return output


def DisparityMap(left, right, window):
    padding = window // 2
    left_img = Pad(left, padding)
    right_img = Pad(right, padding)
    height, width = left_img.shape
    d_map = pb.zeros(left.shape, dtype=float)
    for row in range(height - window + 1):
        for col in range(width - window + 1):
            bestdist = float('inf')
            shift = 0
            left_pixel = left_img[row:row + window, col:col + window]
            l_bound, r_bound, step = Search(
                col, window, width)
            for i in range(l_bound, r_bound, step):
                right_pixel = right_img[row:row + window, i:i + window]
                ssd = pb.sum((left_pixel - right_pixel) ** 2)
                if ssd < bestdist:
                    bestdist = ssd
                    shift = i
            d_map[row, col] = col - shift
    disparity_SGBM = cv2.normalize(d_map, d_map, alpha=255,
                                   beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = pb.uint8(disparity_SGBM)
    return d_map,disparity_SGBM
