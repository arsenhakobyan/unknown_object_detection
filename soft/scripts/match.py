#!/usr/bin/env python3
import numpy as np
import cv2
import sys

def get_slope(point1, point2):
    first = point2[0][0] - point1[0][0]
    second = point2[0][1] - point1[0][1]
    first_arr = np.array([first])
    second_arr = np.array([second])
    ret = cv2.cartToPolar(first_arr, second_arr, angleInDegrees=True)[1]
    return ret

def positiveAngle(angle):
    if 0 > angle:
        angle += 360
    return angle

def isViableQuad(corners_list):
    slopes, angles = [], []
    for i in range(4):
        slopes.append(get_slope(corners_list[i], corners_list[(i+1)%4]))

    for i in range(4):
        angles.append(positiveAngle(slopes[i] - slopes[(i+3)%4]))

    for i in angles:
        if 30 > i or 150 < i:
            return False
    return True

def imgCorners(image):
    image_w, image_h = image.shape[1], image.shape[0]
    return np.float32([[0,0], [image_w,0], [image_w, image_h], [0, image_h]]).reshape(-1,1,2)


def match(hd_image, map_image):
    ratio = 1
    matcher = cv2.FlannBasedMatcher.create()
    sift = cv2.xfeatures2d.SIFT_create(0, 5, 0.02)
    expected_num_matches = 50
    hd_image = cv2.cvtColor(hd_image, cv2.COLOR_BGR2RGB)
    map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
    hd_keypoints, hd_descriptors = sift.detectAndCompute(hd_image, None)
    map_keypoints, map_descriptors = sift.detectAndCompute(map_image, None)
    knn_matches = matcher.knnMatch(hd_descriptors, map_descriptors, 2)

    matches = []
    for i in knn_matches:
        if i[0].distance < i[1].distance * ratio :
            matches.append((i[0].distance / i[1].distance, i[0]))
    matches = sorted(matches, key=lambda i : i[0])
    if 4 > len(matches):
        return
    good_matches = [i[1] for i in matches[:expected_num_matches]]
    obj = np.array([hd_keypoints[good_matches[i].queryIdx].pt for i in range(len(good_matches))])
    scene = np.array([map_keypoints[good_matches[i].trainIdx].pt for i in range(len(good_matches))])

    homography, mask = cv2.findHomography(obj, scene, cv2.RANSAC, 5.)
    returning_dst = []
    if homography.size != 0:
        dst = cv2.perspectiveTransform(imgCorners(hd_image), homography)
        if isViableQuad(dst):
            for i in dst:
                returning_dst.append(i[0])
            return (True, np.array(returning_dst))
    return (False, None)

def print_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x = {x}, y = {y}')

if __name__ == "__main__":
    print ("Start")
    image = cv2.imread(sys.argv[1])

    h, w = image.shape[:2]
    max_x, max_y = h, w
    corners = np.float32([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]])

    map_image = cv2.imread(sys.argv[2])
    pred = match(image, map_image)
    pred = pred[1].astype(int)
    p1 = pred[0]
    p2 = pred[1]
    p3 = pred[2]
    p4 = pred[3]
    color = (255, 0, 0)
    thickness = 2
    cv2.line(map_image, p1, p2, color, thickness)
    cv2.line(map_image, p2, p3, color, thickness)
    cv2.line(map_image, p3, p4, color, thickness)
    cv2.line(map_image, p4, p1, color, thickness)

    cv2.namedWindow('image')

    cv2.setMouseCallback('image', print_coordinates)

    cv2.imshow("image", map_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

