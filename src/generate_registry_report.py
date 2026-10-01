import json,os
p='artifacts/production_model_report.json'
if os.path.exists(p):
 print('Production model report:'); print(json.dumps(json.load(open(p)),indent=4))
else: print('Run the evaluation first.')
