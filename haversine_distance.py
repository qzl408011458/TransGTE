'''
Created on 22 dic 2016

@author: alberto
'''
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

if __name__ == '__main__':
    # LON1 = -8.6912941
    # LAT1 = 41.1383507
    # LON2 = -8.5520090
    # LAT2 = 41.1859353

    LON1, LAT1 = 103.9876, 30.6030
    LON2, LAT2 = 104.1679, 30.7315

    seg_lon = (LON2 - LON1) / 75
    seg_lat = (LAT2 - LAT1) / 140
    print(haversine(LON1, LAT1, LON1 + seg_lon, LAT1))
    print(haversine(LON1, LAT1, LON1, LAT1 + seg_lat))

    print(haversine(LON1 + seg_lon, LAT1, LON1 + seg_lon*2, LAT1))
    print(haversine(LON1, LAT1 + seg_lat, LON1, LAT1 + seg_lat*2))