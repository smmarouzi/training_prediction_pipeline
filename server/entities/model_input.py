from typing import List, Optional
from pydantic import BaseModel

class Features(BaseModel):
    dist_to_restaurant: float
    Hdist_to_restaurant: float
    avg_Hdist_to_restaurants: float
    date_day_number: float
    restaurant_id: int
    Five_Clusters_embedding: int
    h3_index: int
    date_hour_number: float
    restaurants_per_index: int
    
