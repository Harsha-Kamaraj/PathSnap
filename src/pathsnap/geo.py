"""Geodesy helpers.

Longitude/latitude are angles, so Euclidean distance in degrees is wrong
(1 deg of longitude is ~111 km * cos(lat), i.e. ~72 km at 49.8 N but 111 km
of latitude). Each trip covers at most a few tens of km, so we project every
trip into the UTM zone of its own centroid. Within one UTM zone the scale
error is < 0.1 %, which is far below GPS noise (metres), and we can then use
plain Euclidean geometry in metres (fast, and works with shapely / KD-trees).

Haversine is provided as an independent cross-check.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_008.8


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    """lon/lat (WGS84) <-> local UTM metres for one trip."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def to_xy(self, lon, lat):
        return self._fwd.transform(np.asarray(lon), np.asarray(lat))

    def to_lonlat(self, x, y):
        return self._inv.transform(np.asarray(x), np.asarray(y))


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
