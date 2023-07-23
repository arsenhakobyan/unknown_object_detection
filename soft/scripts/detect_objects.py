#!/usr/bin/env python3
import cv2
import numpy as np
import math
import argparse
import os
import glob
from matplotlib import pyplot as plt


def meters_per_pixel(altitude, camera_hres, hfov):
    print ("Hres, hfov: ", camera_hres, hfov)
    # Calculate the real world size of a pixel at a certain altitude
    # Note that this assumes the camera is looking straight down
    #fov_rad = np.deg2rad(fov)
    hfov_rad = math.radians(hfov)
    m_per_pixel = 2 * altitude * math.tan(hfov_rad/2) / camera_hres
    print ("Meter per pixel: ", m_per_pixel)
    return m_per_pixel





def detect_objects(image, target_size, m_per_pixel):

    # Calculate the target area
    target_area = target_size[0] * target_size[1]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("image", hsv)
    cv2.waitKey(0)

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imshow("image", yuv)
    cv2.waitKey(0)


    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate the mean of the pixel values
    mean_val = np.mean(blurred)

    # Set the lower and upper threshold values
    lower = int(max(0, (1.0 - 0.15) * mean_val))
    upper = int(min(255, (1.0 + 0.15) * mean_val))
    # Find edges in the image
    edges = cv2.Canny(blurred, lower, upper)
    cv2.imshow("image", edges)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_area_in_pixels = target_area/(m_per_pixel**2)

    # Iterate over each contour and determine if it approximates to the target size
    i = 0
    for cnt in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate the size of the rectangle in meters
        size = w * m_per_pixel, h * m_per_pixel

        # Calculate the area of the rectangle in meters^2
        #area = size[0] * size[1]
        area = w * h
        #area = cv2.contourArea(cnt)

        ## Compare the size to the target size
        #if area >= target_area * 0.7 and area <= target_area * 0.9:
        #    # This contour is approximately the target size
        #    # Draw the contour and bounding rectangle
        #    cv2.drawContours(image, [cnt], -1, (255, 0, 0), 2)
        #    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Compare the size to the target size
        if abs(area - target_area_in_pixels) < target_area_in_pixels * 0.1:
            # This contour is approximately the target size
            # Draw the contour and bounding rectangle
            ### obj = image[y:y+h, x:x+w]
            ### obj = cv2.resize(obj, (0, 0), fx=3, fy = 3)
            ### cv2.imshow("name_"+str(i), obj)
            ### cv2.waitKey(0)
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ## Compare the size to the target size
        #if area >= target_area * 1.1 and area <= target_area * 1.5:
        #    # This contour is approximately the target size
        #    # Draw the contour and bounding rectangle
        #    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
        #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the image
    cv2.imshow('image', image)
    cv2.waitKey(0)


def split_image(image, n, overlap):
    h, w, _ = image.shape
    tiles = []
    for i in range(0, w, int(w/n)):
        w_dist = w/n
        if i > 0: 
            i -= overlap
            w_dist += overlap
        for j in range (0, h, int(h/n)):
            h_dist = h/n
            if j > 0: 
                j -= overlap
                h_dist += overlap
            tile_y = int(max(0, j))
            tile_h = int(min(j + h_dist, h))
            tile_x = int(max(0, i))
            tile_w = int(min(i + w_dist, w))
            tiles.append([tile_x, tile_y, tile_w, tile_h])
            ### cv2.rectangle(image, (tile_x, tile_y), (tile_w, tile_h), (200, 255, 0), 2)
            ### cv2.imshow("image", image)
            ### cv2.waitKey(0)
    return tiles


def show(image, name = "default", wk = 0):
    cv2.imshow(name, image)
    cv2.waitKey(wk)



def bounding_box_area_with_given_angle(w, h, theta):
    theta_rad = math.radians(theta)
    W = w * abs(math.cos(theta_rad)) + h * abs(math.sin(theta_rad))
    H = w * abs(math.sin(theta_rad)) + h * abs(math.cos(theta_rad))
    return (W * H)

def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def draw_line_on_contour(image, angle, x, y):
    length = 150;
    P1 = (x, y);
    P2 =  ((int)(P1[0] + length * math.cos(angle * np.pi / 180.0)),  (int)(P1[1] + length * math.sin(angle * np.pi / 180.0)))
    cv2.line(image, P1, P2, (0, 255, 0), 1)


def detect_edges(image):
    # Calculate the mean of the pixel values
    mean_val = np.mean(image)
    # Set the lower and upper threshold values
    print (mean_val)
    lower = int(max(0, (1.0 - 0.99) * mean_val))
    upper = int(min(255, (1.0 + 0.75) * mean_val))
    # Find edges in the image
    edges = cv2.Canny(image, lower, upper)
    show(edges)
    return edges


def filter_by_color(image):
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    show(image_hsv)

    # Compute the average color of the image
    average_color = cv2.mean(image_hsv)[:3] # Ignore the fourth value, which is alpha
    print (average_color)

    # Calculate the average value (brightness) of the image
    average_value = np.average(image_hsv[:,:,2])
    
    
    hue = 0
    sat = int(average_color[1] + 20)
    val = int(average_value+1)
    lower = np.array([hue, sat, val])
    upper = np.array([360, 255, 255])

    # Create a mask for pixels with a higher value than the average
    mask = cv2.inRange(image_hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    show (result)
    return result

    # Define the range around the average color
    # Here we specify a range of +/- 10 for each channel
    lower_bound = np.array([max(0, c - 10) for c in average_color])
    upper_bound = np.array([min(255, c + 100) for c in average_color])
    
    # Create a mask of the pixels within the specified range
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    
    # Bitwise-AND the mask with the original image to get an image with just the
    # colors you're interested in
    result = cv2.bitwise_and(image, image, mask=mask)
    show(result)






    
    
    # Define the lower and upper bounds for the color you're interested in
    # For red, the hue can be around 0 - 10 and 170 - 180
    lower_bound_red1 = np.array([0, 10, 120])
    upper_bound_red1 = np.array([10, 255, 255])
    
    lower_bound_red2 = np.array([170, 10, 120])
    upper_bound_red2 = np.array([180, 255, 255])

    # For blue
    lower_bound_blue = np.array([110, 10, 120])
    upper_bound_blue = np.array([130, 255, 255])
    
    # For green
    lower_bound_green = np.array([30, 10, 120])
    upper_bound_green = np.array([90, 255, 255])

    # Create a mask of the pixels within the specified range
    mask1 = cv2.inRange(image_hsv, lower_bound_red1, upper_bound_red1)
    mask2 = cv2.inRange(image_hsv, lower_bound_red2, upper_bound_red2)
    
    # Combine the two masks
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Create a mask of the pixels within the specified range
    mask_blue = cv2.inRange(image_hsv, lower_bound_blue, upper_bound_blue)
    mask_green = cv2.inRange(image_hsv, lower_bound_green, upper_bound_green)
    
    mask = mask_red + mask_blue + mask_green
    
    #mask = cv2.bitwise_or(mask_red, mask_blue)
    #mask = cv2.bitwise_or(mask, mask_green)

    # Bitwise-AND the mask with the original image to get an image with just the 
    # colors you're interested in
    result_red = cv2.bitwise_and(image, image, mask=mask_red)
    result_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    result_green = cv2.bitwise_and(image, image, mask=mask_green)

    result = cv2.bitwise_and(image, image, mask=mask)
    show (result)

    
    ## Show the result
    #show(result_red)
    #show(result_blue)
    #show(result_green)


def morph_erod_dilate(image):

    image = cv2.dilate(image, (5, 5), iterations=2)
    image = cv2.erode(image, (5, 5), iterations=1)
    return image


def detect_objects_cropped(image, target_size_px, target_area_px, max_target_area):
    res_cnt = []
    show(image)

    #tiles = split_image(image, 4, 100)
    #scale = 2 # Cropped tiles' upscaling factor
    #target_area_px = target_area_px * (scale**2)

    #for a in tiles:
    #    x1, y1, w1, h1 = a
    #    image1 = image[y1:h1, x1:w1]

    ####     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ####     show(gray)

    ####     clahe = cv2.createCLAHE(clipLimit=1)
    ####     gray = clahe.apply(gray)
    ####     show(gray)

    ####     #gray = cv2.resize(gray, (0, 0), fx = scale, fy = scale)
    ####     #show(gray)

    ####     # Apply Gaussian Blur
    ####     blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    filtered = filter_by_color(image)
    edges = detect_edges(filtered)
    show (edges)

    #edges = morph_erod_dilate(edges)
    #show (edges)

    # Find contour * 1.1s
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour and determine if it approximates to the target size
    for cnt in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        rect = cv2.minAreaRect(cnt)

        # rect[2] will give the angle of the rectangle
        angle = rect[2]

        # The angle returned by minAreaRect is in the range [-90, 0)
        # As the rectangle rotates clockwise the angle tends toward 0
        # So we can find the angle of rotation by subtracting it from 90

        angle1 = translateRotation(angle, rect[1][0], rect[1][1])

        area_with_angle = bounding_box_area_with_given_angle(target_size_px[1], target_size_px[0], angle)

        area = w * h

        # Compare the size to the target size
        if area > target_area_px * 0.7 and area < area_with_angle * 1.5:
            #cv2.drawContours(image2, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res_cnt.append(cnt)

    ## Show the image
    show(image)

    return res_cnt


def detect_objects_c(original_image, image,  target_size, m_per_pixel):
    # Calculate the target area
    target_area = target_size[0] * target_size[1]

    tiles = split_image(original_image, 4, 100)
    for a in tiles:
        x, y, w, h = a
        print (x, y, w, h)
        image1 = image[y:h, x:w]
        cv2.imshow("image", image1)
        cv2.waitKey(0)
        image1 = cv2.resize(image1, (0, 0), fx=2, fy = 2)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Calculate the mean of the pixel values
        mean_val = np.mean(blurred)


    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Calculate the mean of the pixel values
    mean_val = np.mean(blurred)

    # Set the lower and upper threshold values
    lower = int(max(0, (1.0 - 0.25) * mean_val))
    upper = int(min(255, (1.0 + 0.05) * mean_val))
    # Find edges in the image
    edges = cv2.Canny(blurred, lower, upper)
    cv2.imshow("image", edges)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_area_in_pixels = target_area/(m_per_pixel**2)

    res_cnt = []
    # Iterate over each contour and determine if it approximates to the target size
    for cnt in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        # Compare the size to the target size
        if abs(area - target_area_in_pixels) < target_area_in_pixels * 0.1:
            # This contour is approximately the target size
            # Draw the contour and bounding rectangle
            ### obj = image[y:y+h, x:x+w]
            ### obj = cv2.resize(obj, (0, 0), fx=3, fy = 3)
            ### cv2.imshow("name_"+str(i), obj)
            ### cv2.waitKey(0)
            cv2.drawContours(original_image, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res_cnt.append(cnt)

    ## Show the image
    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    return res_cnt

def get_intersecting_contours(contours1, contours2):
    intersecting_contours = []
    
    for contour1 in contours1:
        for contour2 in contours2:
            # Create filled masks for each contour
            mask1 = np.zeros_like(image)
            mask2 = np.zeros_like(image)
            print (contour1)
            print (contour2)
            cv2.drawContours(mask1, [contour1], -1, (255), thickness=cv2.FILLED)
            cv2.drawContours(mask2, [contour2], -1, (255), thickness=cv2.FILLED)

            # Calculate the intersection between the two masks
            intersection = cv2.bitwise_and(mask1, mask2)

            # If there is any intersection, add the pair of contours to the list
            if np.any(intersection):
                intersecting_contours.append((contour1, contour2))

    return intersecting_contours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to the image")
    parser.add_argument("--dir", help="directory with images")
    parser.add_argument("--altitude", help="altitude of the drone in meters", type=float, default=500)
    parser.add_argument("--ts1", help="target size 1 in meters", type=float)
    parser.add_argument("--ts2", help="target size 2 in meters", type=float)
    args = parser.parse_args()

    # Get a list of all the image files in the directory
    images = glob.glob(os.path.join(args.dir, '*.jpg')) + glob.glob(os.path.join(args.dir, '*.png'))



    # Define the target size (in meters)
    print ("1", images)

    for image_file in images:
        
        print ("image_file: ", image_file)
        # Load the image
        image = cv2.imread(image_file)

        # Calculate meters per pixel at the given altitude
        m_per_pixel = meters_per_pixel(args.altitude, image.shape[1], 62.2)
        target_size_px = (args.ts1 / m_per_pixel, args.ts2 / m_per_pixel)
        target_area_px = target_size_px[0] * target_size_px[1]
        max_bounding_dim = math.sqrt(target_size_px[0]**2 + target_size_px[1]**2)
        max_target_area = max_bounding_dim**2 

        res = detect_objects_cropped(image, target_size_px, target_area_px, max_target_area)


