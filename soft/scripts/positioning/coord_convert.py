#!/usr/bin/env python3

import numpy as np
import wgs84
from pyproj import CRS, Transformer

class CCoordinate: # Coordinate converters
    def __init__(self, lat_ref=0., lng_ref=0., alt_ref=0.):
        self.wgs84_ll_crs = CRS.from_epsg(4326)
        self.update_ref(lat_ref, lng_ref, alt_ref)

    def set_transforms(self, lng):
        utm_zone = round((lng + 180) / 6 + 0.5)
        utm_epsg = 32600 + utm_zone

        self.wgs84_utm_crs = CRS.from_epsg(utm_epsg)
        self.T_ll2utm = Transformer.from_crs(self.wgs84_ll_crs, self.wgs84_utm_crs)
        self.T_utm2ll = Transformer.from_crs(self.wgs84_utm_crs, self.wgs84_ll_crs)

    def update_ref(self, lat_ref, lng_ref, alt_ref):
        self.lat_ref = lat_ref
        self.lng_ref = lng_ref
        self.alt_ref = alt_ref

        self.set_transforms(self.lng_ref)

        lat_ref = np.deg2rad(self.lat_ref)
        lng_ref = np.deg2rad(self.lng_ref)

        self.Tecef2ned = np.array([[-np.sin(lat_ref) * np.cos(lng_ref), -np.sin(lat_ref) * np.sin(lng_ref),  np.cos(lat_ref)],
                                   [-np.sin(lng_ref),                    np.cos(lng_ref),                    0              ],
                                   [-np.cos(lat_ref) * np.cos(lng_ref), -np.cos(lat_ref) * np.sin(lng_ref), -np.sin(lat_ref)]])
        
    def ll2utm(self, lat, lng):
        return self.T_ll2utm.transform(lat, lng)

    def utm2ll(self, m_x, m_y):
        return self.T_utm2ll.transform(m_x, m_y)

    def earthrad(self, lat):
        R_N = wgs84.a / (1 - wgs84._ecc_sqrd * np.sin(lat) ** 2) ** 0.5
        R_M = wgs84.a * (1 - wgs84._ecc_sqrd) / (1 - wgs84._ecc_sqrd * np.sin(lat) ** 2) ** 1.5

        return R_N, R_M
    
    def lla2ecef(self, lat, lng, alt):
        lat = np.deg2rad(lat)
        lng = np.deg2rad(lng)

        Rew, Rns = self.earthrad(lat)

        x = (Rew + alt) * np.cos(lat) * np.cos(lng)
        y = (Rew + alt) * np.cos(lat) * np.sin(lng)
        
        z = ((1 - wgs84._ecc_sqrd) * Rew + alt) * np.sin(lat)

        return np.array([x, y, z])

    def ecef2lla(self, ecef):
        x, y, z = ecef
        lng = np.arctan2(y, x)

        # Iteration to get Latitude and Altitude
        p = np.sqrt(x ** 2 + y ** 2)
        lat = np.arctan2(z, p * (1 - wgs84._ecc_sqrd))
        
        err = 1
        while(np.abs(err) > 1e-10):

            Rew, Rns = self.earthrad(lat)
            h = p / np.cos(lat) - Rew
            
            err = np.arctan2(z * (1 + wgs84._ecc_sqrd * Rew * np.sin(lat) / z), p) - lat
            
            lat = lat + err
        
        lat = np.rad2deg(lat)
        lng = np.rad2deg(lng)

        return np.array([lat, lng, h])

    def ecef2ned(self, ecef):
        ecef = np.array([ecef]).T
        ned = self.Tecef2ned @ ecef
        ned = ned.T[0]

        return ned[0], ned[1], ned[2]

    def ned2ecef(self, ned):
        ned = np.array([ned]).T

        ecef = self.Tecef2ned.T @ ned
        ecef = ecef.T[0]

        return ecef[0], ecef[1], ecef[2]

    def lla2ned(self, lat, lng, alt):

        ecef  = self.lla2ecef(lat, lng, alt)
        ecef0 = self.lla2ecef(self.lat_ref, self.lng_ref, self.alt_ref)

        ned  = self.ecef2ned(ecef - ecef0)

        return np.array(ned)

    def ned2lla(self, ned):
        ecef = self.ned2ecef(ned)
        ecef_ref = self.lla2ecef(self.lat_ref, self.lng_ref, self.alt_ref)

        ecef += ecef_ref

        lla = self.ecef2lla(ecef)
        return lla
