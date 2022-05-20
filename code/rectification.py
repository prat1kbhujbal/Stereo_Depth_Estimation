import numpy as pb
import cv2
import matplotlib.pyplot as plt


def EpipolarLineEndPoints(img, F, p):
    img_width = img.shape[1]
    el = pb.dot(F, pb.array([p[0], p[1], 1]).reshape(3, 1))
    p1 = (0, int((-el[2] / el[1])[0]))
    p2 = (img.shape[1], int(((-img_width * el[0] - el[2]) / el[1])[0]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def ViewEpipolarLines(F, pts1, pts2, img1, img2, figure, title):
    plt.figure(figure)
    plt.title(title)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0]), int(pts1[i][1])
        p1, p2 = EpipolarLineEndPoints(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)
    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0]), int(pts2[i][1])
        p1, p2 = EpipolarLineEndPoints(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)
    ax1.axis('off')
    ax2.axis('off')


def Rectify(p1s, p2s, F, img1, img2):
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        pb.float32(p1s),
        pb.float32(p2s),
        F, imgSize=img1.shape[1:: -1])
    img1_w = cv2.warpPerspective(
        img1, H1, img1.shape[1:: -1])
    img2_w = cv2.warpPerspective(img2, H2, img1.shape[1:: -1])
    img1_copy = img1_w.copy()
    img2_copy = img2_w.copy()

    print('H1 = ', H1)
    print('H2 = ', H2)

    p1s_rectified = cv2.perspectiveTransform(
        p1s.reshape(-1, 1, 2), H1).reshape(-1, 2)
    p2s_rectified = cv2.perspectiveTransform(
        p2s.reshape(-1, 1, 2), H2).reshape(-1, 2)

    H2_inv = pb.linalg.inv(H2.T)
    H1_inv = pb.linalg.inv(H1)
    F_rectified = pb.dot(H2_inv, pb.dot(F, H1_inv))

    for i in range(p1s_rectified.shape[0]):
        cv2.circle(
            img1_copy,
            (int(p1s_rectified[i, 0]),
                int(p1s_rectified[i, 1])),
            8, (255, 0, 0),
            2)
        cv2.circle(
            img2_copy,
            (int(p2s_rectified[i, 0]),
                int(p2s_rectified[i, 1])),
            8, (255, 0, 0),
            2)

    ViewEpipolarLines(
        F_rectified, p2s_rectified[: 100],
        p1s_rectified[: 100],
        img1_copy, img2_copy, 3, "Epipolar Lines- Rectified")

    img1_rectified_reshaped = cv2.cvtColor(img1_w, cv2.COLOR_BGR2GRAY)
    img1_rectified_reshaped = cv2.resize(img1_rectified_reshaped, (600, 400))
    img2_rectified_reshaped = cv2.cvtColor(img2_w, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.resize(img2_rectified_reshaped, (600, 400))
    return img1_rectified_reshaped, img2_rectified_reshaped
