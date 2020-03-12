import cv2
import sys
import numpy as np
import random
from pdf2image import convert_from_path
import img2pdf

MAX_FEATURES = 200
RATIO_ROBUSTNESS=0.70
THRESHOLD_RATIO_INLIERS=0.85
THRESHOLD_REPROJTION_ERROR=3
MAX_NUM_TRIAL=1000

# def find_homo(matches):
#     matrix  = []
#     for i in matches:
#         # print("\ni: ", i, "\n")
#         x, y = [i[0][0], i[0][1]]
#         # print("x: ", x)
#         x_t, y_t = [i[1][0], i[1][1]]

#         matrix_1 = [x, y, 1, 0, 0, 0, -x_t * x, -x_t * y, -x_t]
#         # print("\nmatrix_1: ", matrix_1)
#         matrix_2 = [0, 0, 0, x, y, 1, -y_t * x, -y_t * y, -y_t]
#         matrix.append(matrix_1)
#         matrix.append(matrix_2)
#         # print("\nmatrix: ", matrix, "\n")

#     temp_matrix = np.matrix(matrix)
#     u, s, v = np.linalg.svd(temp_matrix)
#     homo = v[-1, :].reshape(3, 3)
#     homo = homo/v[-1, -1]

#     return homo

# def find_homography_ransac(matches):
#     best_H = None
#     best_ratio = 0

#     # 1000 loops
#     for i in range(MAX_NUM_TRIAL):
#         p1 = matches[random.randint(0, len(matches)-1)]
#         p2 = matches[random.randint(0, len(matches)-1)]
#         p3 = matches[random.randint(0, len(matches)-1)]
#         p4 = matches[random.randint(0, len(matches)-1)]
#         p5 = matches[random.randint(0, len(matches)-1)]
#         p6 = matches[random.randint(0, len(matches)-1)]
#         # print("Loop ", i, "\np1: ", p1, ", p2: ", p2, ", p3: ", p3, ", p4: ", p4)
#         matches = []
#         matches.append(p1)
#         matches.append(p2)
#         matches.append(p3)
#         matches.append(p4)
#         matches.append(p5)
#         matches.append(p6)
#         # print("matches", matches)

#         # matrix that align 2 sets of feature points
#         # reference: http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
#         # page 17 and 18
#         homo = find_homo(matches)

#         inlier_list = []
#         counter = 1
#         # temp = 0
#         for j in matches:
#             # print(counter)
#             p1x, p1y = j[0][0], j[0][1]
#             img_1 = np.dot(homo, [p1x, p1y, 1])
#             img_1 = np.matrix(img_1/img_1.item(2))

#             p2x, p2y = j[1][0], j[1][1]
#             img_2 = np.matrix([p2x, p2y, 1])

#             error = np.linalg.norm(img_1[0:2] - img_2[0:2])
#             counter+=1
#             # print("error: ", error)
#             if error < THRESHOLD_REPROJTION_ERROR:
#                 # print("\nerror * error * error * error: ", error)
#                 inlier_list.append(j)

#         # print("inlier_list", len(inlier_list))
#         actual_ratio = len(inlier_list) / len(matches)
#         # print("actual_ratio: ", actual_ratio)
#         if actual_ratio > THRESHOLD_RATIO_INLIERS:
#             best_H = homo
#             # print("actual_ratio=====>: ", actual_ratio)
#             if best_ratio < actual_ratio:
#                 # print("actual_ratio----------->: ", actual_ratio)
#                 best_ratio = actual_ratio
#                 best_H = find_homo(inlier_list)
#                 # print("000000000000000best_ratio: ", best_ratio)

#     print("homo: \n",  best_H)

#     return best_H
# def extract_and_match_feature(img_1, img_2):
#     gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#     gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp_1, des_1 = sift.detectAndCompute(gray_1, None)
#     kp_2, des_2 = sift.detectAndCompute(gray_2, None)
#     # print("kp_1", len(kp_1))
#     # print("kp_2", len(kp_2))

#     matches = []
#     for i in range(len(des_1)):
#         d1 = [ 9999999999, -1]
#         d2 = [10000000000, -1]
#         for j in range(len(des_2)):
#             # distance = np.sqrt(np.sum(des_1[i] - des_2[j]) * * 2)
#             distance = np.linalg.norm(des_1[i] - des_2[j])

#             if distance < d1[0]:
#                 d2 = d1
#                 d1 = [distance, j]
#                 # print("i: ", i, ", j: ", j, ", \td1: ", d1)
#             elif distance < d2[0]:
#                 d2 = [distance, j]
#                 # print("i: ", i, ", j: ", j, ", \td2: ", d2)
#         # print("d1: ", d1[0], ", d2: ", d2[0])
#         # print("d1/d2: ", d1[0]/d2[0])
#         if d1[0]/d2[0] < RATIO_ROBUSTNESS:
#             p1x, p1y = kp_1[i].pt
#             p2x, p2y = kp_2[d1[1]].pt
#             matches.append([[p1x, p1y], [p2x, p2y]])

#     print("length of matched points:", len(matches))

#     return matches

def find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x, p1y], [p2x, p2y]], ....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None
    best_ratio = 0
    # to be completed ...
    print('\n4. Use RANSAC algorithm to find homography to warp image 1 to align it to image 2...\n')
    # 1000 loops
    for i in range(max_num_trial):
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

    # matrix  = []
    # for i in matches:
        # print("i: ", i)
        # x, y = [i[0][0], i[0][1]]
        # print("x: ", x)
        # print("xy: ", y)
        # x_t, y_t = [i[1][0], i[1][1]]

        # matrix_1 = [0, -x_t, y * x_t]
        # matrix_2 = [x_t, 0, -x * x_t]
        # matrix_3 = [-y * x_t, x * x_t, 0]
        # matrix.append(matrix_1)
        # matrix.append(matrix_2)
        # matrix.append(matrix_3)
        # print("matrix: ", matrix)
        # break

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
        home = v[-1, :].reshape(3, 3)
        home = home/v[-1, -1]
        # print(home)

        inlier_list = []
        counter = 1
        # temp = 0
        for j in list_pairs_matched_keypoints:
            # print(counter)
            p1x, p1y = j[0][0], j[0][1]
            img_1 = np.dot(home, [p1x, p1y, 1])
            img_1 = np.matrix(img_1/img_1.item(2))

            p2x, p2y = j[1][0], j[1][1]
            img_2 = np.matrix([p2x, p2y, 1])

            error = np.linalg.norm(img_1[0:2] - img_2[0:2])
            counter+=1
            # print("error: ", error)
            if error < threshold_reprojtion_error:
                # print("\nerror * error * error * error: ", error)
                inlier_list.append(j)

        # print("inlier_list", len(inlier_list))
        actual_ratio = len(inlier_list) / len(list_pairs_matched_keypoints)
        # print("actual_ratio: ", actual_ratio)
        if actual_ratio > threshold_ratio_inliers:
            best_H = home
            # print("actual_ratio=====>: ", actual_ratio)
            if best_ratio < actual_ratio:
                print("actual_ratio----------->: ", actual_ratio)
                best_ratio = actual_ratio
                matrix  = []
                
                for i in inlier_list:
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
                home = v[-1, :].reshape(3, 3)
                home = home/v[-1, -1]
                best_H = home
                # print("000000000000000best_ratio: ", best_ratio)

    return best_H

def extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x, p1y], [p2x, p2y]]]
    '''
    print('\n2. Extract SIFT features from input image 1 and image 2...\n')
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)
    # print("kp_1", len(kp_1))
    # print("kp_2", len(kp_2))

    print('\n3. Bruteforce search to find a list of pairs of matched feature points...\n')
    list_pairs_matched_keypoints = []

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
        if d1[0]/d2[0] < ratio_robustness:
            p1x, p1y = kp_1[i].pt
            p2x, p2y = kp_2[d1[1]].pt
            list_pairs_matched_keypoints.append([[p1x, p1y], [p2x, p2y]])

    # index = 0
    # for i in range(len(des_1)):
        # d1 = 0
        # d2 = 0
        # for j in range(len(des_2)):
        #     # distance = np.sum(des_1[i] - des_2[j]) * * 2
        #     distance = np.linalg.norm(des_1[i] - des_2[j])
        #     if d1 == 0:
        #         d1 = distance
        #     elif distance < d1:
        #         d2 = d1
        #         d1 = distance
        #         index = j
        #     elif distance < d2:
        #         d2 = distance
        # print("d1: ", d1, ", d2: ", d2)
        # print("d1/d2: ", d1/d2)
        # if d1/d2 < ratio_robustness:
        #     print("Hit!!!!!!!!!!!!!!!!!!!!!!")
        #     p1x, p1y = kp_1[i].pt
        #     p2x, p2y = kp_2[index].pt
        #     list_pairs_matched_keypoints.append([[p1x, p1y], [p2x, p2y]])

    # print("list_pairs_matched_keypoints: \n", list_pairs_matched_keypoints)
    print("length of matched points(must be 77):", len(list_pairs_matched_keypoints), '\n')

    return list_pairs_matched_keypoints

def warp_image(img_1, H_1, img_2):
    h, w, _ = img_2.shape
    result = cv2.warpPerspective(img_1, H_1, (w, h))

    return result

def align_orb_homo(img_1, img_2):
    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    kp_1, des_1 = orb.detectAndCompute(gray_1, None)
    kp_2, des_2 = orb.detectAndCompute(gray_2, None)

    # find matches
    match_list = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = match_list.match(des_1, des_2, None)

    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)

    # keep good matches
    good = int(len(matches) * RATIO_ROBUSTNESS)
    matches = matches[:good]

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
    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)

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

    return binary, binaryinv

if __name__ == '__main__':
    # read images
    img_1 = cv2.imread("input.png", cv2.IMREAD_COLOR)

    pages = convert_from_path('reference.pdf', size=(805,876))
    for page in pages:
        page.save('reference.png', 'PNG')

    # align image basic on the reference
    img_2 = cv2.imread("reference.png", cv2.IMREAD_COLOR)

    # orb features
    result1 = align_orb_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output_sift.png", result1)

    # sift features
    result1 = align_sift_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output_orb.png", result1)

    # burforce
    result2 = align_my_homo(img_1, img_2)
    # save image
    cv2.imwrite("output/output1.png", result2)

    binary, binaryinv= enhance(result2)
    cv2.imwrite("output/output1_binary.png", binary)
    cv2.imwrite("output/output1_binaryinv.png", binaryinv)

    with open("output/output.pdf","wb") as f:
	    f.write(img2pdf.convert('output/output1.png'))
