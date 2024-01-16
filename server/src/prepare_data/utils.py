import pandas as pd
import numpy as np
import collections
from math import radians 
from src.prepare_data.config import Radius

def copy_df(df:pd.DataFrame):
    """Creates a copy of the DataFrame for further use"""

    return df.copy()

def calculate_eucleadian_dist(p1x:float, p1y:float, p2x:float, p2y:float):
    p1 = (p2x - p1x)**2
    p2 = (p2y - p1y)**2
    dist = np.sqrt(p1 + p2)
    return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

def calculate_avg_distance_to_resturants(courier_lat:float, courier_lon:float, restaurants_ids:dict):
    return np.mean([calculate_eucleadian_dist(v['lat'], v['lon'], \
        courier_lat, courier_lon) for v in restaurants_ids.values()])

def calculate_haversine_dist(lat1:float, lon1:float, lat2:float, lon2:float, radians_metric = "kilometers"):
   
    radius = Radius(radians_metric)
    
    if isinstance(lat1, collections.abc.Sequence):
        dLat = np.array([radians(l2 - l1) for l2,l1 in zip(lat2, lat1)])
        dLon = np.array([radians(l2 - l1) for l2,l1 in zip(lon2, lon1)])
        lat1 = np.array([radians(l) for l in lat1])
        lat2 = np.array([radians(l) for l in lat2])
    else:
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    dist = radius.R*c
    return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist


def calculate_avg_Hdist_to_restaurants(courier_lat,courier_lon, restaurants_ids:dict):
  return np.mean([calculate_haversine_dist(v['lat'], v['lon'], \
      courier_lat, courier_lon) for v in restaurants_ids.values()])
  
  
