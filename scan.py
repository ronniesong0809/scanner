import cv2
import sys
import numpy as np
import random
from pdf2image import convert_from_path
import img2pdf

MAX_FEATURES = 200
RATIO_ROBUSTNESS=0.75
THRESHOLD_RATIO_INLIERS=0.85
THRESHOLD_REPROJTION_ERROR=3
MAX_NUM_TRIAL=1000

def find_homography_ransac(list_pairs_matched_keypoints):
    best_H = None
    best_ratio = 0

    # 1000 loops
    for i in range(MAX_NUM_TRIAL):
        p1 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        p2 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        p3 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        p4 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        p5 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        p6 = list_pairs_matched_keypoints[random.randint(0, len(list_pairs_matched_keypoints)-1)]
        # print("Loop ", i, "\np1: ", p1, ", p2: ", p2, ", p3: ", p3, ", p4: ", p4)
        matches = []
        matches.append(p1)
        matches.append(p2)
        matches.append(p3)
        matches.append(p4)
        matches.append(p5)
        matches.append(p6)
        # print("matches", matches)

        # matrix that align 2 sets of feature points
        # reference: http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
        # page 17 and 18
        matrix  = []
        for i in matches:
            # print("\ni: ", i, "\n")
            x, y = [i[0][0], i[0][1]]
            # print("x: ", x)
            x_t, y_t = [i[1][0], i[1][1]]

            matrix_1 = [x, y, 1, 0, 0, 0, -x_t * x, -x_t * y, -x_t]
            # print("\nmatrix_1: ", matrix_1)
            matrix_2 = [0, 0, 0, x, y, 1, -y_t * x, -y_t * y, -y_t]
            matrix.append(matrix_1)
            matrix.append(matrix_2)
            # print("\nmatrix: ", matrix, "\n")

        temp_matrix = np.matrix(matrix)
        u, s, v = np.linalg.svd(temp_matrix)
        homo = v[-1, :].reshape(3, 3)
        homo = homo/v[-1, -1]

        inlier_list = []
        counter = 1
        # temp = 0
        for j in list_pairs_matched_keypoints:
            # print(counter)
            p1x, p1y = j[0][0], j[0][1]
            img_1 = np.dot(homo, [p1x, p1y, 1])
            img_1 = np.matrix(img_1/img_1.item(2))

            p2x, p2y = j[1][0], j[1][1]
            img_2 = np.matrix([p2x, p2y, 1])

            error = np.linalg.norm(img_1[0:2] - img_2[0:2])
            counter+=1
            # print("error: ", error)
            if error < THRESHOLD_REPROJTION_ERROR:
                # print("\nerror * error * error * error: ", error)
                inlier_list.append(j)

        # print("inlier_list", len(inlier_list))
        actual_ratio = len(inlier_list) / len(matches)
        # print("actual_ratio: ", actual_ratio)
        if actual_ratio > THRESHOLD_RATIO_INLIERS:
            best_H = homo
            # print("actual_ratio=====>: ", actual_ratio)
            if best_ratio < actual_ratio:
                # print("actual_ratio----------->: ", actual_ratio)
                best_ratio = actual_ratio
                matrix  = []
                for i in matches:
                    # print("\ni: ", i, "\n")
                    x, y = [i[0][0], i[0][1]]
                    # print("x: ", x)
                    x_t, y_t = [i[1][0], i[1][1]]

                    matrix_1 = [x, y, 1, 0, 0, 0, -x_t * x, -x_t * y, -x_t]
                    # print("\nmatrix_1: ", matrix_1)
                    matrix_2 = [0, 0, 0, x, y, 1, -y_t * x, -y_t * y, -y_t]
                    matrix.append(matrix_1)
                    matrix.append(matrix_2)
                    # print("\nmatrix: ", matrix, "\n")

                temp_matrix = np.matrix(matrix)
                u, s, v = np.linalg.svd(temp_matrix)
                homo = v[-1, :].reshape(3, 3)
                homo = homo/v[-1, -1]
                best_H = homo
                # print("000000000000000best_ratio: ", best_ratio)

    print("homo: \n",  best_H)

    return best_H
def extract_and_match_feature(img_1, img_2):
    print("\nextract SIFT feature from input image and reference image")

    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)
    # print("kp_1", len(kp_1))
    # print("kp_2", len(kp_2))

    matches = []
    for i in range(len(des_1)):
        d1 = [ 9999999999, -1]
        d2 = [10000000000, -1]
        for j in range(len(des_2)):
            # distance = np.sqrt(np.sum(des_1[i] - des_2[j]) * * 2)
            distance = np.linalg.norm(des_1[i] - des_2[j])

            if distance < d1[0]:
                d2 = d1
                d1 = [distance, j]
                # print("i: ", i, ", j: ", j, ", \td1: ", d1)
            elif distance < d2[0]:
                d2 = [distance, j]
                # print("i: ", i, ", j: ", j, ", \td2: ", d2)
        # print("d1: ", d1[0], ", d2: ", d2[0])
        # print("d1/d2: ", d1[0]/d2[0])
        if d1[0]/d2[0] < RATIO_ROBUSTNESS:
            p1x, p1y = kp_1[i].pt
            p2x, p2y = kp_2[d1[1]].pt
            matches.append([[p1x, p1y], [p2x, p2y]])

    print("length of matched points:", len(matches))

    return matches

def warp_image(img_1, H_1, img_2):
    h, w, _ = img_2.shape
    result = cv2.warpPerspective(img_1, H_1, (w, h))

    return result

def drawKeypoints(img_1, img_2, gray_1, gray_2, kp_1, kp_2):
    kp_sift_1 = cv2.drawKeypoints(gray_1, kp_1, img_1)
    kp_sift_2 = cv2.drawKeypoints(gray_2, kp_2, img_2)

    cv2.imwrite("output/kp/kp_1.png", kp_sift_1)
    cv2.imwrite("output/kp/kp_2.png", kp_sift_2)

def align_orb_homo(img_1, img_2):
    print("\nextract ORB feature from input image and reference image")

    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    kp_1, des_1 = orb.detectAndCompute(gray_1, None)
    kp_2, des_2 = orb.detectAndCompute(gray_2, None)

    # drawKeypoints(img_1, img_2, gray_1, gray_2, kp_1, kp_2)

    # find matches
    match_list = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = match_list.match(des_1, des_2, None)

    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)

    # keep good matches
    good = int(len(matches) * RATIO_ROBUSTNESS)
    matches = matches[:good]

    print("length of matched points:", len(matches))

    # draw matches
    image = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches, None)
    cv2.imwrite("output/matches_orb.jpg", image)

    # extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp_1[match.queryIdx].pt
        points2[i, :] = kp_2[match.trainIdx].pt

    # find homography
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("homo: \n",  homography)

    # backward mapping
    h, w, _ = img_2.shape
    result = cv2.warpPerspective(img_1, homography, (w, h))

    return result

def align_sift_homo(img_1, img_2):
    print("\nextract SIFT feature from input image and reference image")

    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)

    # drawKeypoints(img_1, img_2, gray_1, gray_2, kp_1, kp_2)

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_1, des_2, k=2)

    # keep good matches
    good = []
    good1 = []
    for m,n in matches:
        if m.distance < RATIO_ROBUSTNESS*n.distance:
            good.append([m])
            good1.append(m)

    print("length of matched points:", len(good))

    # draw matches
    h, w, _ = img_2.shape
    image = np.zeros((h, w))
    image = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good, None, flags=2)
    cv2.imwrite("output/matches_sift.jpg", image)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # for i, match in enumerate(good1):
    #     points1[i, :] = kp_1[match.queryIdx].pt
    #     points2[i, :] = kp_2[match.trainIdx].pt

    points1 = np.float32([kp_1[m.queryIdx].pt for m in good1]).reshape(-1, 1, 2)
    points2 = np.float32([kp_2[m.trainIdx].pt for m in good1]).reshape(-1, 1, 2)

    # find homography
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("homo: \n",  homography)

    # backward mapping
    result = cv2.warpPerspective(img_1, homography, (w, h))

    return result

def align_my_homo(img_1, img_2):
    matches = extract_and_match_feature(img_1, img_2)

    H_1 = find_homography_ransac(matches)

    wrap = warp_image(img_1, H_1, img_2)

    return wrap

def enhance(img):
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    # [0,-1,0],[-1,5,-1],[0,-1,0]
    binaryinv = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    return binary

if __name__ == '__main__':
    # read images
    img_1 = cv2.imread("input.png", cv2.IMREAD_COLOR)
    h, w, _ = img_1.shape
    print("h: ", h, ", w: ", w)

    pages = convert_from_path('reference.pdf', size=(w,h))
    for page in pages:
        page.save('reference.png', 'PNG')

    # align image basic on the reference
    img_2 = cv2.imread("reference.png", cv2.IMREAD_COLOR)

    # orb features
    result_orb = align_orb_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output_orb.png", result_orb)

    # sift features
    result_sift = align_sift_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output_sift.png", result_sift)

    # sift features
    result2 = align_my_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output1.png", result2)

    binary = enhance(result2)
    cv2.imwrite("output/output1_binary.png", binary)

    with open("output/output1.pdf","wb") as f:
	    f.write(img2pdf.convert('output/output1.png'))
