import subprocess,sys
scripts=['src/validate_data.py','src/preprocess_pipeline.py','src/validate_outputs.py']
for script in scripts:
 print(f'\n[INFO] ---> Executing {script}...'); result=subprocess.run([sys.executable,script])
 if result.returncode: raise SystemExit(f'[ERROR] Pipeline halted at {script}')
print('\n[SUCCESS] Lab 5 Production Pipeline fully executed!')
