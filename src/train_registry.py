import numpy as np,mlflow,mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def train_and_register_model():
    print('[INFO] Starting MLflow Registry run...'); mlflow.set_experiment('Life_Expectancy_Prediction')
    Xtr=np.load('data/processed/X_train_final.npy'); Xte=np.load('data/processed/X_test_final.npy'); ytr=np.load('data/processed/y_train.npy'); yte=np.load('data/processed/y_test.npy')
    params={'n_estimators':200,'max_depth':15,'random_state':42,'n_jobs':-1}
    with mlflow.start_run(run_name='RandomForest_Registry_V1'):
        mlflow.log_params(params); model=RandomForestRegressor(**params).fit(Xtr,ytr); pred=model.predict(Xte)
        metrics={'mae':mean_absolute_error(yte,pred),'rmse':np.sqrt(mean_squared_error(yte,pred)),'r2':r2_score(yte,pred)}; mlflow.log_metrics(metrics)
        mlflow.log_artifact('models/preprocessor.pkl',artifact_path='preprocessing_pipeline')
        mlflow.sklearn.log_model(model,name='random_forest_model',registered_model_name='Life_Expectancy_Production_Model')
        print('[SUCCESS] Model registered as Life_Expectancy_Production_Model')
if __name__=='__main__': train_and_register_model()
