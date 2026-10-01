import subprocess,sys
for s in ['src/validate_data.py','src/preprocess_pipeline.py','src/train.py','src/evaluate.py']:
 print(f'\n[INFO] ---> {s}'); r=subprocess.run([sys.executable,s]);
 if r.returncode: raise SystemExit(1)
