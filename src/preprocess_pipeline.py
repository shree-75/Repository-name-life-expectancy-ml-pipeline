import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET='Life expectancy '
DATA_PATH='data/raw/Life Expectancy Data.csv'

def run_preprocessing():
    print('[INFO] Starting Sklearn Pipeline Preprocessing...')
    df=pd.read_csv(DATA_PATH)
    # Remove rows with an unknown target; feature missing values are handled by the pipeline.
    df=df.dropna(subset=[TARGET]).copy()
    X=df.drop(columns=[TARGET])
    y=df[TARGET].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    cat_cols=X_train.select_dtypes(include=['object','category']).columns.tolist()
    num_cols=X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat_pipeline=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    preprocessor=ColumnTransformer([('num',num_pipeline,num_cols),('cat',cat_pipeline,cat_cols)])
    print('[INFO] Fitting and transforming training data...')
    X_train_final=preprocessor.fit_transform(X_train)
    X_test_final=preprocessor.transform(X_test)
    os.makedirs('data/processed',exist_ok=True); os.makedirs('models',exist_ok=True)
    np.save('data/processed/X_train_final.npy',X_train_final)
    np.save('data/processed/X_test_final.npy',X_test_final)
    np.save('data/processed/y_train.npy',y_train.to_numpy(dtype=np.float64))
    np.save('data/processed/y_test.npy',y_test.to_numpy(dtype=np.float64))
    joblib.dump(preprocessor,'models/preprocessor.pkl')
    metadata={'dataset_name':'WHO Life Expectancy Dataset','target':TARGET,'target_description':'Life expectancy in years','rows_after_target_cleaning':int(len(df)),'train_shape':list(X_train_final.shape),'test_shape':list(X_test_final.shape),'numerical_features':num_cols,'categorical_features':cat_cols}
    with open('data/processed/dataset_metadata.json','w') as f: json.dump(metadata,f,indent=4)
    print('[SUCCESS] Preprocessing completed. Pipeline saved as preprocessor.pkl.')

if __name__=='__main__': run_preprocessing()
