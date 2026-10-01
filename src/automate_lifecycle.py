import subprocess,sys
steps=['src/validate_data.py','src/preprocess_pipeline.py','src/validate_outputs.py','src/train.py','src/evaluate.py','src/validate_reproducibility.py']
for s in steps:
 print(f'\n[INFO] ---> Executing {s}...'); r=subprocess.run([sys.executable,s]);
 if r.returncode: raise SystemExit(f'[ERROR] Lifecycle halted at {s}')
print('\n[SUCCESS] Full ML lifecycle completed!')
