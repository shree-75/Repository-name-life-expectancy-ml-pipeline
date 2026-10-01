import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def train_and_predict():
    Xtr=np.load('data/processed/X_train_final.npy'); Xte=np.load('data/processed/X_test_final.npy'); ytr=np.load('data/processed/y_train.npy'); yte=np.load('data/processed/y_test.npy')
    model=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    pred=model.predict(Xte); metrics={'mae':mean_absolute_error(yte,pred),'rmse':np.sqrt(mean_squared_error(yte,pred)),'r2':r2_score(yte,pred)}
    return pred,metrics

def validate_reproducibility():
    print('\n--- Reproducibility Validation ---'); p1,m1=train_and_predict(); p2,m2=train_and_predict()
    pm=np.allclose(p1,p2); mm=all(np.isclose(m1[k],m2[k]) for k in m1)
    print('Predictions identical :',pm); print('Metrics identical     :',mm)
    if pm and mm: print('[SUCCESS] Reproducibility validation passed!'); print(f"MAE: {m1['mae']:.4f} | RMSE: {m1['rmse']:.4f} | R²: {m1['r2']:.4f}")
    else: raise SystemExit('[ERROR] Reproducibility validation failed!')
if __name__=='__main__': validate_reproducibility()
