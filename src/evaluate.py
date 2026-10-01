import os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET='Life expectancy '

def run_evaluation():
    print('Starting Model Evaluation...')
    X_test=np.load('data/processed/X_test_final.npy'); y_test=np.load('data/processed/y_test.npy')
    model=joblib.load('models/random_forest_baseline.pkl')
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred); rmse=np.sqrt(mean_squared_error(y_test,y_pred)); r2=r2_score(y_test,y_pred)
    print('\n--- Model Evaluation Report ---')
    print(f'MAE  : {mae:.4f} years'); print(f'RMSE : {rmse:.4f} years'); print(f'R²   : {r2:.4f}'); print('-------------------------------\n')
    os.makedirs('outputs',exist_ok=True); os.makedirs('artifacts',exist_ok=True)
    df=pd.read_csv('data/raw/Life Expectancy Data.csv').dropna(subset=[TARGET]).copy()
    _, X_test_raw=train_test_split(df.drop(columns=[TARGET]),test_size=0.2,random_state=42)
    errors=X_test_raw.reset_index(drop=True); errors['Actual_Life_Expectancy']=y_test; errors['Predicted_Life_Expectancy']=y_pred; errors['Absolute_Error']=np.abs(errors['Actual_Life_Expectancy']-errors['Predicted_Life_Expectancy'])
    errors.sort_values('Absolute_Error',ascending=False).to_csv('outputs/prediction_errors.csv',index=False)
    metrics={'mae':float(mae),'rmse':float(rmse),'r2':float(r2)}
    with open('artifacts/production_model_report.json','w') as f: import json; json.dump(metrics,f,indent=4)
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(y_test,y_pred,alpha=.55); lo=min(y_test.min(),y_pred.min()); hi=max(y_test.max(),y_pred.max()); ax.plot([lo,hi],[lo,hi],linestyle='--'); ax.set_xlabel('Actual Life Expectancy'); ax.set_ylabel('Predicted Life Expectancy'); ax.set_title('Actual vs Predicted Life Expectancy'); fig.savefig('artifacts/actual_vs_predicted.png',bbox_inches='tight'); plt.close(fig)
    print('Evaluation complete and error analysis saved!')
if __name__=='__main__': run_evaluation()
