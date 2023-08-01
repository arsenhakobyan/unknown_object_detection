
# Table 3.1: WGS 84  Four Defining Parameters
a = 6378137.0 # Semi-major Axis [m]
f = 1./298.257223563 # Flattening
omega_E = 7292115.0e-11 # Angular velocity of the Earth [rad/s]
omega_E_GPS = 7292115.1467e-11 # Angular velocity of the Earth [rad/s]
                              # According to ICD-GPS-200

GM = 3986004.418e8 # Earth's Gravitational Constant [m^3/s^2]
                   # (mass of earth's atmosphere included)

GM_GPS = 3986005.0e8 # The WGS 84 GM value recommended for GPS receiver usage 
                     # by the GPS interface control document (ICD-GPS-200) 
                     # differs from the current refined WGS 84 GM value.
                     #
                     # Details for this difference can be read in the WGS84 
                     # reference: 3.2.3.2 "Special Considerations for GPS"

# Table 3.3: WGS 84 Ellipsoid Derived Geometric Constants
_b = 6356752.3142 # Semi-minor axis [m]
_ecc = 8.1819190842622e-2 # First eccentricity
_ecc_sqrd = 6.69437999014e-3 # First eccentricity squared
