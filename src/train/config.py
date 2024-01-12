class Data:
    train_size = 0.77
    test_size = 0.33
    split_random_state=42
    features = ['dist_to_restaurant', 'Hdist_to_restaurant', 'avg_Hdist_to_restaurants',\
        'date_day_number', 'restaurant_id', 'Five_Clusters_embedding', 'h3_index',\
            'date_hour_number', 'restaurants_per_index']
    target = ['orders_busyness_by_h3_hour']
class RF:
    max_depth = 5
    regr_random_state = 0
    n_jobs = -1
    min_samples_leaf = 50 
    n_estimators = 150 
    best_params = {
        "max_depth": 5,
        "min_samples_leaf": 50,
        "n_estimators": 150 
    }
    
class GridSearch:
    params = {
        "max_depth": [4,5],
        "min_samples_leaf": [50, 70],
        "n_estimators": [100, 150] 
    }
    cv = 3
    verbose = 1
    n_jobs = -1
    scoring = "r2"
    
    