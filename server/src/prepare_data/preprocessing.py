import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import h3 

from src.prepare_data.utils import copy_df
from src.prepare_data.config import Centroids, Random, H3, FinalFeatures
from src.prepare_data.utils import calculate_eucleadian_dist
from src.prepare_data.utils import calculate_haversine_dist
from src.prepare_data.utils import calculate_avg_distance_to_resturants
from src.prepare_data.utils import calculate_avg_Hdist_to_restaurants

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

def prepare_restaurants_ids(df:pd.DataFrame) -> dict:
    """Add resturant id column"""
    logger.info('PREPARING RESURANT IDs...')
    
    #unique restaurants
    restaurants_ids = {}
    for a,b in zip(df.restaurant_lat, df.restaurant_lon):
        id = "{}_{}".format(a,b)
        restaurants_ids[id] = {"lat": a, "lon":b}
    for i,key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]['id'] = i
    
    return restaurants_ids

def drop_nan(df:pd.DataFrame):
    """Drops nan rows"""
    logger.info('DROPPING THE NAN ROWS...')
    
    df.dropna(axis = 0, inplace = True)

    return df

def add_restaurants_ids(df:pd.DataFrame):
    """Add resturant id column"""
    logger.info('ADDING RESURANT ID COLUMN...')
    
    restaurants_ids = prepare_restaurants_ids(df)
    df['restaurant_id']=[restaurants_ids["{}_{}".format(a,b)]['id'] for a,b in zip(df.restaurant_lat, df.restaurant_lon)]
    
    return df

def assign_centroids(df):
    """Assign centroids"""
    
    logger.info('ASSIGNING CENTROIDS...')
    
    restaurants_ids = prepare_restaurants_ids(df)
    np.random.seed(Random.seed)
    df_restaurants = pd.DataFrame([{"lat": v['lat'], "lon": v['lon']} for v in restaurants_ids.values()])
    centroids = df_restaurants.sample(Centroids.K)

    assignation = []
    assign_errors = []
    centroids_list = [c for i,c in centroids.iterrows()]
    for i,obs in df.iterrows():
        # Estimate error
        all_errors = [calculate_eucleadian_dist(centroid['lat'],
                                centroid['lon'],
                                obs['courier_lat'],
                                obs['courier_lon']) for centroid in centroids_list]

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.min(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.min(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)
    df['Five_Clusters_embedding'] =assignation
    df['Five_Clusters_embedding_error'] =assign_errors
    return df

def add_distance_columns(df:pd.DataFrame):
    """add different courier distance measurements to resturants columns"""
    
    restaurants_ids = prepare_restaurants_ids(df)
    logger.info('ADDING 4 DISTANCE COLUMNS...')
    df['dist_to_restaurant'] = calculate_eucleadian_dist(df.courier_lat, df.courier_lon, df.restaurant_lat, df.restaurant_lon)
    logger.info('EUCLEADIAN DISTANCE ADDED...')
    df['avg_dist_to_restaurants'] = [calculate_avg_distance_to_resturants(lat, lon, restaurants_ids) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    logger.info('AVG DISTANCE ADDED...')
    df['Hdist_to_restaurant'] = calculate_haversine_dist(df.courier_lat.tolist(), df.courier_lon.tolist(), df.restaurant_lat.tolist(), df.restaurant_lon.tolist())
    logger.info('HDIST ADDED...')
    df['avg_Hdist_to_restaurants'] = [calculate_avg_Hdist_to_restaurants(lat, lon, restaurants_ids) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    logger.info('AVG HDIST ADDED...')
    
    return df 

def add_datetime_columns(df:pd.DataFrame):
    """convert datetime format and add day, hour columns"""
    
    logger.info('FIXING DATETIME FORMAT...')
    df['courier_location_timestamp']=  pd.to_datetime(df['courier_location_timestamp'], errors='coerce')
    df['order_created_timestamp'] = pd.to_datetime(df['order_created_timestamp'], errors='coerce')
    
    logger.info('ADDING DATETIME FORMAT...')
    df['date_day_number'] = [d for d in df.courier_location_timestamp.dt.day_of_year]
    df['date_hour_number'] = [d for d in df.courier_location_timestamp.dt.hour]
    
    return df 

def add_h3_clustering(df:pd.DataFrame):
    """add h3 clustering"""
    logger.info('ADDING H3 CLUSTERING...')    
    
    df['h3_index'] = [h3.geo_to_h3(lat,lon, H3.resolution) for (lat, lon) in zip(df.courier_lat, df.courier_lon)]
    return df
    
def add_orders_busyness(df:pd.DataFrame):
    """add orders busyness by h3 hour"""
    logger.info('ADDING ORDERS BUSYNESS BY H3...')    
    
    index_list = [(i,d,hr) for (i,d,hr) in zip(df.h3_index, df.date_day_number, df.date_hour_number)]

    set_indexes = list(set(index_list))
    dict_indexes = {label: index_list.count(label) for label in set_indexes}
    df['orders_busyness_by_h3_hour'] = [dict_indexes[i] for i in index_list]
    return df 

def add_restaurants_per_index(df):
    """add resturant per index"""
    logger.info('ADDING RESTAURANTS PER INDEX...') 
    restaurants_counts_per_h3_index = {a:len(b) for a,b in zip(df.groupby('h3_index')['restaurant_id'].unique().index, df.groupby('h3_index')['restaurant_id'].unique()) }
    df['restaurants_per_index'] = [restaurants_counts_per_h3_index[h] for h in df.h3_index]
    return df

def convert_to_correct_dtype(df:pd.DataFrame):
    """Corrects the dtype of the column h3_index"""

    logger.info('CONVERTING DTYPE OF h3_index COLUMN TO category...')
    df['h3_index'] = df.h3_index.astype('category')

    return df
 
def encoder(df:pd.DataFrame):
    """encode to categorical columns"""
    
    logger.info('ENCODING...')
    
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    
    return df

def feature_selection(df:pd.DataFrame):
    """select final features for training"""
    
    logger.info('FEATURE SELECTION...')
    df = df[FinalFeatures.features + FinalFeatures.target]
    return df
    

def data_preprocessing_pipeline(df:pd.DataFrame):

    df = df.pipe(copy_df)\
        .pipe(drop_nan)\
        .pipe(add_restaurants_ids)\
        .pipe(add_distance_columns)\
        .pipe(assign_centroids)\
        .pipe(add_datetime_columns)\
        .pipe(add_h3_clustering)\
        .pipe(add_orders_busyness)\
        .pipe(add_restaurants_per_index)\
        .pipe(convert_to_correct_dtype)\
        .pipe(encoder)\
        .pipe(drop_nan)\
        .pipe(feature_selection)
    return df