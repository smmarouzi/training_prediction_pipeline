from src.prepare_data.preprocessing import *
from src.prepare_data.config import Radius
from src.train.model import *
import pandas as pd 

if __name__ == "__main__":
    """ 
    data_path = 'assignment/final_dataset.csv'
    train_df = pd.read_csv(data_path)
    train_df_preprocessed = data_preprocessing_pipeline(train_df)
    prepared_data_path = "assignment/model_data.csv"
    train_df_preprocessed.to_csv(prepared_data_path, index=False)
    """
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    prepared_data_path = "assignment/model_data.csv"
    df = pd.read_csv(prepared_data_path)
    df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)
    train_path = "assignment/train.csv"
    test_path = "assignment/test.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    import pandas as pd
    import pickle
    from src.train.model import BusynessEstimation
    from src.train.config import Data
    from src.utils.utils import get_image_data

    train_path = "assignment/train.csv"
    test_path = "assignment/test.csv"
    
    # Read train and test data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Instantiate the model class
    busyness_model = BusynessEstimation(
                                        test_data.copy()
                                        )
                                        
    # Create X_train and y_train
    X_train = train_data.drop(Data.target, axis=1)
    y_train = train_data[Data.target]

    # Fit the model (training pipeline consists of feature engineering, feature selection and training an xgboost model)
    busyness_model.fit(X_train, y_train)
    
    # Save the best hyperparameters as an artifact
    print(busyness_model.best_params)
    
    model_path = "assignment/rf.sav"
    # Save the model as an artifact
    with open(model_path, 'wb') as f: 
        pickle.dump({
            "pipeline": busyness_model.model_pipeline,
            "target": busyness_model.target,
            "scores_dict": busyness_model.scores}, f)
    """
    filename = "assignment/rf.sav"
    import pickle
    p = pickle.load(open(filename, 'rb'))
    print(p["target"])
    print(p["scores_dict"])
    print(p["pipeline"])
    test_df = pd.read_csv("assignment/test.csv")
    X_test = test_df.drop(Data.target, axis=1)
    Y_test = test_df[Data.target[0]]
    result = p["pipeline"].score(X_test, Y_test)
    print(result)
    
    