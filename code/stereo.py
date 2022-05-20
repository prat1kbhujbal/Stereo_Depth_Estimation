import cv2
import argparse
import numpy as pb
import matplotlib.pyplot as plt
from calibration import *
from rectification import *
from correspondence import *
from computedepthimage import *


def main():
    '''Main Function'''
    # Arguments
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--filePath', default='../data_files/',
        help='Images path')
    parse.add_argument(
        "--dataset",
        default=3,
        help="Dataset Number")
    Args = parse.parse_args()
    file_path = Args.filePath
    dataset = int(Args.dataset)

    if dataset == 1:
        image1 = cv2.imread(file_path + 'curule/im0.png')
        image2 = cv2.imread(file_path + 'curule/im1.png')
        kc1 = pb.array(
            [[1758.23, 0, 977.42],
             [0, 1758.23, 552.15],
             [0, 0, 1]])
        kc2 = pb.array(
            [[1758.23, 0, 977.42],
             [0, 1758.23, 552.15],
             [0, 0, 1]])
        fl, bl = kc1[0, 0], 88.39
    if dataset == 2:
        image1 = cv2.imread(file_path + 'octagon/im0.png')
        image2 = cv2.imread(file_path + 'octagon/im1.png')
        kc1 = pb.array(
            [[1742.11, 0, 804.90],
             [0, 1742.11, 541.22],
             [0, 0, 1]])
        kc2 = pb.array(
            [[1742.11, 0, 804.90],
             [0, 1742.11, 541.22],
             [0, 0, 1]])
        fl, bl = kc1[0, 0], 221.76
    if dataset == 3:
        image1 = cv2.imread(file_path + 'pendulum/im0.png')
        image2 = cv2.imread(file_path + 'pendulum/im1.png')
        kc1 = pb.array(
            [[1729.05, 0, -364.24],
             [0, 1729.05, 552.22],
             [0, 0, 1]])
        kc2 = pb.array(
            [[1729.05, 0, -364.24],
             [0, 1729.05, 552.22],
             [0, 0, 1]])
        fl, bl = kc1[0, 0], 537.75

    img1_cpy = image1.copy()
    img2_cpy = image2.copy()

    pt1, pt2 = Features(img1_cpy, img2_cpy)

    best_F, p1s, p2s = GetInlierRANSANC(pt1, pt2)

    ViewEpipolarLines(
        best_F, p2s[: 100],
        p1s[: 100],
        img1_cpy, img2_cpy, 2, "Epipolar Lines-Unrectified")

    E = EssentialMatrixFromFundamentalMatrix(kc1, kc2, best_F)

    r, c = ExtractCameraPose(E)

    pts3ds = Triangulation(r, c, kc1, kc2, p1s, p2s)

    R, C, pt3D = DisambiguateCameraPose(r, c, pts3ds)
    
    print("R =", R)
    print("C =", C)
    
    
    
    # img1_rectified, img2_rectified = Rectify(
    #     p1s, p2s, best_F, img1_cpy, img2_cpy)

    # disparity_img, disparity_imgn = DisparityMap(
    #     img1_rectified, img2_rectified, 3)
    # plt.figure(4)
    # plt.title('Disparity Gray')
    # plt.imshow(disparity_imgn, cmap='gray')
    # plt.figure(5)
    # plt.title('Disparity Heat Map')
    # plt.imshow(disparity_imgn, cmap='hot')

    # depth = DepthMap(bl, fl, disparity_img)

    # plt.figure(6)
    # plt.title('Depth Gray')
    # plt.imshow(depth, cmap='gray')
    # plt.colorbar()
    # plt.figure(7)
    # plt.title('Depth Heat Map')
    # plt.imshow(depth, cmap='hot')
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
