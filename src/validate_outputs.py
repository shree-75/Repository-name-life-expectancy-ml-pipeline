import os,json,numpy as np
def validate_preprocessing_outputs():
    print('[INFO] Validating Preprocessing Outputs...')
    try:
        Xtr=np.load('data/processed/X_train_final.npy'); Xte=np.load('data/processed/X_test_final.npy'); ytr=np.load('data/processed/y_train.npy'); yte=np.load('data/processed/y_test.npy')
    except FileNotFoundError as e: print('[ERROR]',e); return False
    errors=[]
    if np.isnan(Xtr).sum()>0 or np.isnan(Xte).sum()>0: errors.append('NaNs detected after preprocessing.')
    if Xtr.shape[1]!=Xte.shape[1]: errors.append('Feature mismatch between train and test.')
    if len(ytr)!=Xtr.shape[0] or len(yte)!=Xte.shape[0]: errors.append('Row mismatch between features and targets.')
    report={'validation_status':'PASSED' if not errors else 'FAILED','matrix_dimensions':{'X_train_shape':list(Xtr.shape),'X_test_shape':list(Xte.shape),'y_train_shape':list(ytr.shape),'y_test_shape':list(yte.shape)},'data_quality':{'missing_values_X_train':int(np.isnan(Xtr).sum()),'missing_values_X_test':int(np.isnan(Xte).sum())},'errors':errors}
    os.makedirs('artifacts',exist_ok=True)
    with open('artifacts/preprocessing_summary_report.json','w') as f: json.dump(report,f,indent=4)
    if errors:
        print('[ERROR] Output Validation FAILED!'); [print(' -',e) for e in errors]; return False
    print('[SUCCESS] Output Validation PASSED.'); return True
if __name__=='__main__': raise SystemExit(0 if validate_preprocessing_outputs() else 1)
