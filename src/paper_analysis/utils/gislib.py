"""Geographic distance utilities used by spatial and network analyses."""

import numpy as np


def point_euclidean_dist(lon1, lat1, lon2, lat2):
    """
    Calculate the euclidean between two points. The coordinates of the positionfixes in the yh dataset are the grid centroids, and no geographic distances have to be computed; in order to speed up the implementation, the Manhattan distances are therefore defined.
    :param lon_1: The longitude of the first point.
    :param lat_1: The latitude of the first point.
    :param lon_2: The longitude of the second point.
    :param lat_2: The latitude of the second point.
    :return:   The absolute distance between two points
    """
    lon1 = np.array(lon1)
    lat1 = np.array(lat1)
    lon2 = np.array(lon2)
    lat2 = np.array(lat2)

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def point_haversine_dist(lon1, lat1, lon2, lat2, earthradius=6371000):
    # convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    m = earthradius * c
    return m


