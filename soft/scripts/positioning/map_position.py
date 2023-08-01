#!/usr/bin/env python3

from PIL import Image, ExifTags
import cv2
import os
import sys
import numpy as np
from coord_convert import CCoordinate
from matcher import Detector
from utils import logger, dump_error
import matplotlib.pyplot as plt
import argparse

from osgeo import gdal

class MapPosition(CCoordinate):

    def __init__(self, map_filename, scale=5):
        super(MapPosition, self).__init__()
        
        self.map_filename = map_filename
        self.ds = gdal.Open(map_filename)

        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize
        self.matcher = Detector()
        self.scale = scale
        self.count = 0

    def cropped_map2d_pixel(self, x, y):
        xy = np.array([x, y, 1])
        new_xy = self.InvM @ xy
        new_xy //= new_xy[2]

        new_xy *= self.scale
        new_xy += self.scale // 2
        if new_xy[0] > self.max_x or new_xy[1] > self.max_y or new_xy[0] < 0 or new_xy[1] < 0:
            logger.warning(
                '{}, {} pixels of cropped map aren\'t mapping into drone image\'s pixels.')
        return new_xy[0], new_xy[1]

    def d_pixel2cropped_map(self, x, y):
        x /= self.scale
        y /= self.scale

        xy = np.array([x, y, 1])
        new_xy = self.M @ xy
        new_xy //= new_xy[2]
        return new_xy[0], new_xy[1]

    def utm2pixel(self, gt, m_x, m_y):
        xoffset, px_w, rot1, yoffset, rot2, px_h = gt

        m_x -= xoffset
        m_y -= yoffset

        A = np.array([[px_w, rot1], [rot2, px_h]])
        b = np.array([m_x, m_y])

        pixel_xy = np.linalg.solve(A, b)

        pixel_x, pixel_y = int(pixel_xy[0]), int(pixel_xy[1])
        return pixel_x, pixel_y

    def pixel2utm(self, gt, pixel_x, pixel_y):
        xoffset, px_w, rot1, yoffset, rot2, px_h = gt

        m_x = px_w * pixel_x + rot1 * pixel_y + xoffset
        m_y = rot2 * pixel_x + px_h * pixel_y + yoffset

        # shift to the center of the pixel
        m_x += px_w / 2.0
        m_y += px_h / 2.0

        return m_x, m_y

    def pixel2ll(self, x, y):
        pixel_x, pixel_y = self.d_pixel2cropped_map(x, y)
        m_x, m_y = self.pixel2utm(self.cropped_image_gt, pixel_x, pixel_y)
        lat, lng = self.utm2ll(m_x, m_y)
        return lat, lng

    def ll2pixel(self, lat, lng):
        m_x, m_y = self.ll2utm(lat, lng)
        gt = self.ds.GetGeoTransform()
        pixel_x, pixel_y = self.utm2pixel(gt, m_x, m_y)
        return pixel_x, pixel_y

    def crop_map(self, lat, lng, x_size, y_size):
        pixel_x, pixel_y = self.ll2pixel(lat, lng)
        left_x = max(pixel_x - x_size, 0)
        top_y = max(pixel_y - y_size, 0)
        crop_width, crop_height = min(
            2 * x_size, self.width), min(2 * y_size, self.height)
        os.makedirs(os.path.join(os.path.dirname(
            self.map_filename), 'cropped_map'), exist_ok=True)

        filename = os.path.join(os.path.dirname(
            self.map_filename), 'cropped_map', '{}_{}.tif'.format(lat, lng))
        raster_data_cropped = gdal.Translate(filename, self.ds,
                                             srcWin=[left_x, top_y, crop_width, crop_height])
        return raster_data_cropped, filename

    def get_latlng_from_exif(self, info):
        self.exif_table = {}

        for tag, value in info.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            self.exif_table[decoded] = value

        gps_info = {}

        for key in self.exif_table['GPSInfo'].keys():
            decode = ExifTags.GPSTAGS.get(key, key)
            gps_info[decode] = self.exif_table['GPSInfo'][key]
        gps_lat = gps_info['GPSLatitude']
        latitude = float(gps_lat[0]) + \
            (float(gps_lat[1]) / 60) + \
            (float(gps_lat[2]) / 3600)
        if gps_info['GPSLatitudeRef'] == 'S':
            latitude *= -1

        gps_lng = gps_info['GPSLongitude']
        longitude = float(gps_lng[0]) + \
            (float(gps_lng[1]) / 60) + \
            (float(gps_lng[2]) / 3600)
        if gps_info['GPSLongitudeRef'] == 'W':
            longitude *= -1
        return latitude, longitude

    def on_click(self, event):
        self.count += 1
        x = event.xdata
        y = event.ydata
        lat, lng = self.pixel2ll(x, y)
        if event.inaxes is not None:
            print(f'\n{self.count}: You clicked on pixel coordinates: x={int(x)}, y={int(y)}, lat={lat}, lng={lng}\n')

    def on_scroll(self, event):
        ax = plt.gca()
        zoom_factor = 1.1
        if event.button == 'up':
            ax.set_xlim([x / zoom_factor for x in ax.get_xlim()])
            ax.set_ylim([y / zoom_factor for y in ax.get_ylim()])
        elif event.button == 'down':
            ax.set_xlim([x * zoom_factor for x in ax.get_xlim()])
            ax.set_ylim([y * zoom_factor for y in ax.get_ylim()])
        plt.draw()

    def run_matching(self, filename):
        try:
            d_image = Image.open(filename)
            h, w = d_image.size
            self.max_x, self.max_y = h // self.scale, w // self.scale
            self.corners = np.float32(
                [[0, 0], [self.max_x, 0], [self.max_x, self.max_y], [0, self.max_y]])

            resized_d_image = np.array(d_image.resize((self.max_x, self.max_y)))

            d_lat, d_lng = self.get_latlng_from_exif(
                d_image._getexif())

            self.update_ref(d_lat, d_lng, 0)

            cropped_image_dataset, cropped_map_file_name = self.crop_map(
                d_lat, d_lng, self.max_x, self.max_y)

            map_image = np.array(Image.open(cropped_map_file_name))

            pred = self.matcher.run_detector( resized_d_image, map_image)

            if pred[0]:
                self.match = pred[1]
                self.cropped_image_dataset = cropped_image_dataset
                self.cropped_image_gt = self.cropped_image_dataset.GetGeoTransform()

                self.M = cv2.getPerspectiveTransform(self.corners, self.match)
                self.InvM = cv2.getPerspectiveTransform(
                    self.match, self.corners)

                fig, ax = plt.subplots()
                ax.imshow(resized_d_image, aspect='equal')
                plt.axis('off')  # To hide axis values

                # Remove padding and margin
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                fig.canvas.mpl_connect('button_press_event', self.on_click)
                fig.canvas.mpl_connect('scroll_event', self.on_scroll)
                plt.show()

                return cropped_map_file_name
            else:
                logger.warning("NO FOUND ANY MATCHING BETWEEEN UPLOADED IMAGE '{}' AND THE GENERATED MAP '{}'".format(filename, self.map_filename))
        except:
            dump_error()
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map and match images with optional scaling.")
    parser.add_argument("--map", required=True, type=str, help="Path to the map image.")
    parser.add_argument("--image", required=True, type=str, help="Path to the image to be matched.")
    parser.add_argument("--scale", type=int, default=10, help="Scale value used to scale the image before matching. Default is 10.")
    args = parser.parse_args()
    m = MapPosition(args.map, args.scale)
    m.run_matching(args.image)
