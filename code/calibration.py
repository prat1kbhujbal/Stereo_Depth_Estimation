import cv2
import numpy as pb
import matplotlib.pyplot as plt


def Features(image1, image2):
    img1_copy = image1.copy()
    img2_copy = image2.copy()
    image1_gray = cv2.cvtColor(img1_copy, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(img2_copy, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, f1 = orb.detectAndCompute(image1_gray, None)
    kp2, f2 = orb.detectAndCompute(image2_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    points = bf.match(f1, f2)
    matches = sorted(points, key=lambda x: x.distance)
    match_img = cv2.drawMatches(
        img1_copy, kp1, img2_copy, kp2, matches[: 1000], None)
    plt.figure(1)
    plt.title('Matches')
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    pt1 = pb.float32([kp1[i.queryIdx].pt for i in matches])
    pt2 = pb.float32([kp2[j.trainIdx].pt for j in matches])
    return pt1, pt2


def EstimateFundamentalMatrix(p1n, p2n, s1, s2):
    A = pb.zeros((len(p1n), 9))
    for i, (u, v) in enumerate(zip(p1n, p2n)):
        A[i, 0] = u[0] * v[0]
        A[i, 1] = u[1] * v[0]
        A[i, 2] = v[0]
        A[i, 3] = u[0] * v[1]
        A[i, 4] = u[1] * v[1]
        A[i, 5] = v[1]
        A[i, 6] = u[0]
        A[i, 7] = u[1]
        A[i, 8] = 1
    U, S, VT = pb.linalg.svd(A, full_matrices=True)
    F = VT[-1, :]
    F = F.reshape(3, 3)
    u, d, vt = pb.linalg.svd(F)
    d = pb.diag(d)
    d[2, 2] = 0
    Fs = pb.dot(u, pb.dot(d, vt))
    F_cleaned = pb.dot(s2.T, pb.dot(Fs, s1))
    return F_cleaned


def Normalize(pt):
    pts_mean = pb.mean(pt, axis=0)
    x_ = pts_mean[0]
    y_ = pts_mean[1]
    x_r, y_r = pt[:, 0] - x_, pt[:, 1] - y_
    s = (2 / pb.mean(x_r**2 + y_r**2))**0.5
    dm = pb.diag([s, s, 1])
    tm = pb.array([[1, 0, -x_], [0, 1, -y_], [0, 0, 1]])
    ftm = pb.dot(dm, tm)
    hs = pb.column_stack((pt, pb.ones(len(pt))))
    pt_n = (ftm.dot(hs.T)).T
    return pt_n, ftm


def ComputeRANSANCLoss(pts1, pts2, F):
    u = pb.asarray([pts1[0], pts1[1], 1])
    v = pb.asarray([pts2[0], pts2[1], 1])
    error = pb.dot(v.T, pb.dot(F, u))
    error = pb.abs(error)
    return error


def GetInlierRANSANC(pt1, pt2):
    indices = pb.arange(pt1.shape[0])
    best_F = None
    final_idx = []
    rows = pt1.shape[0]
    inliers = 0
    for i in range(2000):
        indices_t = []
        pb.random.shuffle(indices)
        indices_8 = indices[:8]
        p1n, s1 = Normalize(pt1[indices_8])
        p2n, s2 = Normalize(pt2[indices_8])
        F = EstimateFundamentalMatrix(p1n, p2n, s1, s2)
        # print(F)
        for j in range(rows):
            error = ComputeRANSANCLoss(
                pt1[j], pt2[j], F)
            if error < 0.001:
                indices_t.append(j)
        if len(indices_t) > inliers:
            inliers = len(indices_t)
            best_F = F
            p1s = pt1[indices_t]
            p2s = pt2[indices_t]
    print('Best fundamental matrix = {}'.format(best_F))
    return best_F, p1s, p2s


def EssentialMatrixFromFundamentalMatrix(k1, k2, F):
    E = k2.T.dot(F).dot(k1)
    u, s, vt = pb.linalg.svd(E)
    s = [1, 1, 0]
    E = pb.dot(u, pb.dot(pb.diag(s), vt))
    return E


def ExtractCameraPose(E):
    u, d, vt = pb.linalg.svd(E)
    w = pb.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c1 = u[:, 2]
    c2 = -u[:, 2]
    r1 = u.dot(w).dot(vt.T)
    r2 = u.dot(w.T).dot(vt)
    r = [r1, r1, r2, r2]
    c = [c1, c2, c1, c2]
    for i in range(4):
        if (pb.linalg.det(r[i]) < 0):
            r[i] = -r[i]
            c[i] = -c[i]
    return r, c


def Triangulation(r, c, kc1, kc2, p1s, p2s):
    pts3Ds = []
    P1 = kc1 @ pb.hstack((pb.eye(3), pb.zeros((3, 1))))
    for i in range(len(r)):
        P2 = kc2 @ pb.hstack((r[i], -r[i] @ c[i].reshape(3, 1)))
        pts3D = cv2.triangulatePoints(P1, P2, p1s.T, p2s.T)
        pts3Ds.append(pts3D)
    return pts3Ds


def DisambiguateCameraPose(Rs, Cs, pts3ds):
    p_indx = 0
    pt_valid = 0
    for i, (r, c, pt3d) in enumerate(zip(Rs, Cs, pts3ds)):
        n = 0
        c = c.reshape(-1)
        r3 = r[2, :]
        pt3d = pt3d[i]
        for x in pt3d:
            c1p = (x - c)[2]
            c2p = pb.dot(x - c, r3)
            if c1p > 0 and c2p > 0:
                n += 1
        if n > pt_valid:
            pt_valid = n
            p_indx = i
    return Rs[p_indx], Cs[p_indx], pts3ds[p_indx]
