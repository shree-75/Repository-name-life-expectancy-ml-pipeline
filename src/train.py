import numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor

def run_training():
    print('Starting Model Training...')
    X=np.load('data/processed/X_train_final.npy'); y=np.load('data/processed/y_train.npy')
    model=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
    model.fit(X,y)
    joblib.dump(model,'models/random_forest_baseline.pkl')
    print('Model training complete and saved to disk!')
if __name__=='__main__': run_training()
