#!/usr/bin/env python3

from PIL import Image, ExifTags, ImageDraw
import cv2
import os
import sys
import numpy as np
from matcher import Detector
from utils import logger, dump_error
import matplotlib.pyplot as plt
import argparse
import math
import struct
import requests
from bs4 import BeautifulSoup
import pyproj

from osgeo import gdal

class MapPosition:

    def __init__(self, map_filename, scale=5, m_octave_layers=6, m_expected_num_matches=200):
        super(MapPosition, self).__init__()
        
        self.map_filename = map_filename
        self.ds = gdal.Open(map_filename)
        
        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize
        self.matcher = Detector(num_of_octave_layers=m_octave_layers, expected_num_matches=m_expected_num_matches)
        self.scale = scale
        self.count = 0
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.sk42_28408 = pyproj.CRS("EPSG:28408")

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

    def pixel2ll(self, x, y):
        pixel_x, pixel_y = self.d_pixel2cropped_map(x, y)
        xoffset, px_w, rot1, yoffset, rot2, px_h = self.cropped_image_gt

        lat = px_h * pixel_y + rot1 * pixel_x + yoffset
        lng = rot2 * pixel_y + px_w * pixel_x + xoffset

        # shift to the center of the pixel
        lat += px_h / 2.0
        lng += px_w / 2.0

        return lat, lng

    def ll2pixel(self, lat, lng):
        xoffset, px_w, rot1, yoffset, rot2, px_h = self.ds.GetGeoTransform()

        lat -= yoffset
        lng -= xoffset

        A = np.array([[px_h, rot1], [rot2, px_w]])
        b = np.array([lat, lng])

        pixel_xy = np.linalg.solve(A, b)

        pixel_y, pixel_x = int(pixel_xy[0]), int(pixel_xy[1])
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


    def get_elevation(self, lat, lng):
        evf = "./Armenia_Elevation/N" + str(math.floor(lat)) + "E0" + str(math.floor(lng)) + ".hgt";
        with open(evf, 'rb') as hgt:
            size = 1201
            row = int((lat - int(lat)) * (size - 1))
            col = int((lng - int(lng)) * (size - 1))

            pos = ((size - row) * size + col) * 2  # 2 bytes per data point
            hgt.seek(pos)
            elev = struct.unpack('>h', hgt.read(2))[0]
            return elev


    def get_conversion(self, lat, lon):
        base_url = "https://sk42.org/?direction=toSK42"
        full_url = f"{base_url}&convstr={lat}%2C+{lon}"
        response = requests.get(full_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            res = soup.get_text(strip=True)
            start_pos = res.find("SK-42 (XY)")
            end_pos = res.find("Original coordinates")
            res = res[start_pos:end_pos].strip()
            start_pos1 = res.find("X:")

            sk42 = res[start_pos1:end_pos].strip()
            sk42 = sk42.replace("Y:", "\tY:")
            return "SK42: " + sk42
        else:
            return "Response Status is not OK"

    def on_click(self, event):
        self.count += 1
        x = event.xdata
        y = event.ydata
        lat, lng = self.pixel2ll(x, y)
        alt = self.get_elevation(lat, lng)

        if event.inaxes is not None:
            print(f'\n{self.count}: You clicked on pixel coordinates: x,y={int(x)}, {int(y)}\n')
            print("WGS84 Latitude Longitude Altitude: {}, {}, {}".format(lat, lng, alt))
            print(f"SK42.org\t\t ", self.get_conversion(lat, lng))

            transformer_epsg28408 = pyproj.Transformer.from_crs(self.wgs84, self.sk42_28408)

            lat_sk42, lon_sk42 = transformer_epsg28408.transform(lat, lng)
            print(f"SK42 CRS: EPSG:28408\t  X: {lat_sk42}, Y: {lon_sk42}")


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
            self.max_x, self.max_y = (int)(h // self.scale), (int)(w // self.scale)
            self.corners = np.float32(
                [[0, 0], [self.max_x, 0], [self.max_x, self.max_y], [0, self.max_y]])

            resized_d_image = np.array(d_image.resize((self.max_x, self.max_y)))

            d_lat, d_lng = self.get_latlng_from_exif(
                d_image._getexif())


            cropped_image_dataset, cropped_map_file_name = self.crop_map(
                d_lat, d_lng, self.max_x, self.max_y)
            
            print("cropped_map_file_name", cropped_map_file_name)
            map_image = np.array(Image.open(cropped_map_file_name))

            pred = self.matcher.run_detector(resized_d_image, map_image)

            if pred[0]:
                self.match = pred[1]
                cropped_map = Image.open(cropped_map_file_name)
                draw = ImageDraw.Draw(cropped_map)

                line_color = (255, 0, 0)  # Red
                    
                for i in range(3):
                    draw.line([tuple(pred[1][i].astype(int)), tuple(pred[1][i+1].astype(int))], fill=line_color, width=3)
                plt.imshow(cropped_map)

                self.cropped_image_dataset = cropped_image_dataset
                self.cropped_image_gt = self.cropped_image_dataset.GetGeoTransform()

                self.M = cv2.getPerspectiveTransform(self.corners, self.match)
                self.InvM = cv2.getPerspectiveTransform(
                    self.match, self.corners)

                fig, ax = plt.subplots()
                ax.imshow(d_image, aspect='equal')
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
    parser.add_argument("--image", "-i", required=True, type=str, help="Path to the image to be matched.")
    parser.add_argument("--map", default="lernapat_map/map.tif", type=str, help="Path to the map image.")
    parser.add_argument("--scale", type=float, default=2.941, help="Scale value used to scale the image before matching. Default is 2.941.")
    parser.add_argument("--octave-layers", "-ol",  type=int, default=6, help="Number of SIFT octave layers. Default is 6.")
    parser.add_argument("--number-matches", "-nm",  type=int, default=200, help="Number of expected matches. Default is 200.")

    args = parser.parse_args()
    m = MapPosition(args.map, args.scale, args.octave_layers, args.number_matches)
    m.run_matching(args.image)
