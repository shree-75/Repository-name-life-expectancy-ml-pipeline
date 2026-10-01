import os,numpy as np,joblib,mlflow,mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def train_and_track(run_name='RandomForest_Baseline',params=None):
    if params is None: params={'n_estimators':100,'max_depth':10,'random_state':42,'n_jobs':-1}
    Xtr=np.load('data/processed/X_train_final.npy'); Xte=np.load('data/processed/X_test_final.npy'); ytr=np.load('data/processed/y_train.npy'); yte=np.load('data/processed/y_test.npy')
    mlflow.set_experiment('Life_Expectancy_Prediction')
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params); mlflow.log_param('model_family','RandomForestRegressor')
        model=RandomForestRegressor(**params).fit(Xtr,ytr); pred=model.predict(Xte)
        metrics={'mae':mean_absolute_error(yte,pred),'rmse':np.sqrt(mean_squared_error(yte,pred)),'r2':r2_score(yte,pred)}; mlflow.log_metrics(metrics)
        if os.path.exists('data/processed/dataset_metadata.json'): mlflow.log_artifact('data/processed/dataset_metadata.json',artifact_path='metadata')
        mlflow.sklearn.log_model(model,name='model'); joblib.dump(model,'models/random_forest_model.pkl')
        print(f"Metrics logged: MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f}")
if __name__=='__main__': train_and_track()
