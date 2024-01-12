class Radius:
    def __init__(self, metric = "kilometers") -> None:
        if metric == "kilometers":
            self.R = 6372.8 
        elif metric == "mile":
            self.R = 3959.87433
            
class Centroids:
    K = 5
    
class Random:
    seed = 1
    
class H3:
    resolution = 7
    
class FinalFeatures:
    features = ['dist_to_restaurant', 'Hdist_to_restaurant', 'avg_Hdist_to_restaurants',\
        'date_day_number', 'restaurant_id', 'Five_Clusters_embedding', 'h3_index',\
            'date_hour_number', 'restaurants_per_index']
    target = ['orders_busyness_by_h3_hour']