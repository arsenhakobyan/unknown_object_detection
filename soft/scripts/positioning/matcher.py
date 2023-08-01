import numpy as np
import cv2

class Detector:
    def __init__(self, num_of_descriptors=0, num_of_octave_layers=5, thresh=0.02, ratio=0.8, expected_num_matches=50):
        self.matcher = cv2.FlannBasedMatcher.create()
        self.sift = cv2.SIFT_create(num_of_descriptors, num_of_octave_layers, thresh)
        self.ratio = ratio
        self.expected_num_matches = expected_num_matches


    def run_detector(self, hd_image, map_image):
        hd_image = cv2.cvtColor(hd_image, cv2.COLOR_BGR2RGB)
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        hd_keypoints, hd_descriptors = self.sift.detectAndCompute(hd_image, None)
        map_keypoints, map_descriptors = self.sift.detectAndCompute(map_image, None)
        knn_matches = self.matcher.knnMatch(hd_descriptors, map_descriptors, 2)

        matches = []
        for i in knn_matches:
            if i[0].distance < i[1].distance * self.ratio :
                matches.append((i[0].distance / i[1].distance, i[0]))
        matches = sorted(matches, key=lambda i : i[0])
        if 4 > len(matches):
            return
        good_matches = [i[1] for i in matches[:self.expected_num_matches]]
        obj = np.array([hd_keypoints[good_matches[i].queryIdx].pt for i in range(len(good_matches))])
        scene = np.array([map_keypoints[good_matches[i].trainIdx].pt for i in range(len(good_matches))])

        homography, mask = cv2.findHomography(obj, scene, cv2.RANSAC, 5.)
        returning_dst = []
        if homography.size != 0:
            dst = cv2.perspectiveTransform(Detector.imgCorners(hd_image), homography)
            if Detector.isViableQuad(dst):
                for i in dst:
                    returning_dst.append(i[0])
                return (True, np.array(returning_dst))
        return (False, None)

    @staticmethod
    def get_slope(point1, point2):
        first = point2[0][0] - point1[0][0]
        second = point2[0][1] - point1[0][1]
        first_arr = np.array([first])
        second_arr = np.array([second])
        ret = cv2.cartToPolar(first_arr, second_arr, angleInDegrees=True)[1]
        return ret

    @staticmethod
    def positiveAngle(angle):
        if 0 > angle:
            angle += 360
        return angle

    @staticmethod
    def isViableQuad(corners_list):
        slopes, angles = [], []
        for i in range(4):
            slopes.append(Detector.get_slope(corners_list[i], corners_list[(i+1)%4]))

        for i in range(4):
            angles.append(Detector.positiveAngle(slopes[i] - slopes[(i+3)%4]))

        for i in angles:
            if 30 > i or 150 < i:
                return False
        return True

    @staticmethod
    def imgCorners(image):
        image_w, image_h = image.shape[1], image.shape[0]
        return np.float32([[0,0], [image_w,0], [image_w, image_h], [0, image_h]]).reshape(-1,1,2)
