#!/usr/bin/env python3
from multiprocessing import Process, Manager, Value
from PIL import Image
from math import modf
import math
import numpy as np
import time
from pymavlink import mavutil
import cv2
from exiftool import ExifTool
import piexif
import os


def state_define(drone_state, stop_capture):
    mav = mavutil.mavlink_connection('udp:localhost:14551', baudrate=115200)


    while not stop_capture.value:
        msg = mav.recv_match()
        if msg is None:
            continue
        if msg.get_type() == "BAD_DATA":
            continue
        if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
            continue
        elif msg.get_type() == "ATTITUDE":
            drone_state["roll"]  = msg.roll * 180 / np.pi
            drone_state["pitch"] = msg.pitch * 180 / np.pi
            drone_state["yaw"]   = msg.yaw * 180 / np.pi

        elif msg.get_type() == "GLOBAL_POSITION_INT":    
            drone_state["latitude"] = msg.lat / 1e7
            drone_state["longitude"] = msg.lon / 1e7
            drone_state["altitude"] = msg.alt / 1000

        elif msg.get_type() == "LOCAL_POSITION_NED":
            speed = math.sqrt(msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2)
            drone_state["is_stopped"] = speed < 0.5

        elif msg.get_type() == "HEARTBEAT":
            if msg.base_mode == 0:
                continue
            print("Base Mode is", msg.base_mode)
            drone_state["is_armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) 

def format_gps_data(data):
    degrees = int(data)
    minutes = int(modf(data)[0] * 60)
    seconds = int(((modf(data)[0] * 60) - minutes) * 60 * 100)

    return (degrees, 1), (minutes, 1), (seconds, 100)

def add_exif_data(img_path, lat, lon, alt, yaw, pitch, roll, output_path=None):
    # Open the image
    image = Image.open(img_path)

    # Extract the EXIF data if present, else initialize an empty dictionary
    exif_data = image._getexif()
    if exif_data is None:
        exif_data = {}

    # Prepare GPS metadata
    gps_data = {}

    # Set latitude and its reference (N/S)
    if lat >= 0:
        ref = "N"
    else:
        ref = "S"
        lat = -lat
    gps_data[piexif.GPSIFD.GPSLatitudeRef] = ref
    gps_data[piexif.GPSIFD.GPSLatitude] = format_gps_data(lat)

    # Set longitude and its reference (E/W)
    if lon >= 0:
        ref = "E"
    else:
        ref = "W"
        lon = -lon
    gps_data[piexif.GPSIFD.GPSLongitudeRef] = ref
    gps_data[piexif.GPSIFD.GPSLongitude] = format_gps_data(lon)

    #gps_data[piexif.GPSIFD.GPSAltitude] = alt
    #gps_data[piexif.GPSIFD.GPSIMUPitch] = pitch
    #gps_data[piexif.GPSIFD.GPSIMURoll] = roll
    #gps_data[piexif.GPSIFD.GPSIMUYaw] = yaw
    # Check if the image has EXIF data
    if "exif" in image.info:
        exif_dict = piexif.load(image.info["exif"])
    else:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}}

    # Inject GPS metadata into the image's EXIF data
    exif_dict["GPS"] = gps_data
    print(exif_dict)
    exif_bytes = piexif.dump(exif_dict)

    # Save the modified image
    if output_path is None:
        output_path = img_path
    image.save(output_path, "JPEG", exif=exif_bytes, quality=100)

def capture_write(drone_state, stop_capture, camera="piv2"):
    os.makedirs("test_flight", exist_ok=True)

    if camera=="ardu":
        dispW=3840
        dispH=2160
        flip=0
        camSet=f'nvarguscamerasrc ! video/x-raw(memory:NVMM), width={dispW}, height={dispH}, format=NV12, framerate=30/1 !  nvvidconv flip-method={flip} ! video/x-raw, width={dispW}, height={dispH}, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
    elif camera=="piv2":
        dispW=3264
        dispH=2464
        flip=0
        camSet=f'nvarguscamerasrc ! video/x-raw(memory:NVMM), width={dispW}, height={dispH}, format=NV12, framerate=21/1 !  nvvidconv flip-method={flip} ! video/x-raw, width={dispW}, height={dispH}, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'

    cam = cv2.VideoCapture(camSet)

    while not drone_state['is_armed']:
        time.sleep(0.5)
    i = 0
    while drone_state["is_armed"]:
        ret, frame = cam.read()
        if not ret:
            print ("Can not capture")
            continue

        img_name = "test_flight/img_" + str(time.time()) + ".jpg"
        cv2.imwrite(img_name, frame)
        print("For image '{}' by id: {} data is {}".format(img_name, i, drone_state))
        i += 1
        add_exif_data(img_name, drone_state["latitude"], drone_state["longitude"], drone_state["altitude"],
                                drone_state["yaw"], drone_state["pitch"], drone_state["roll"])

    stop_capture.value = True

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":  # confirms that the code is under main function

    with Manager() as manager:

      drone_state = manager.dict({
			"roll":None,
			"pitch":None,
			"yaw":None,
			"latitude":0,
			"longitude":0,
			"altitude":0.,
			"is_armed":False,
            "is_stopped":None
			})

      stop_capture = Value('b', False)

      proc1 = Process(target=state_define, args =(drone_state,stop_capture))
      proc2 = Process(target=capture_write, args =(drone_state,stop_capture))
      
      proc1.start()
      proc2.start()
      
      proc1.join()
      proc2.join()
