import subprocess,sys
r=subprocess.run([sys.executable,'src/train_registry.py']); raise SystemExit(r.returncode)
