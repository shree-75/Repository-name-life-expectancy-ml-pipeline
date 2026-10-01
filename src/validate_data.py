import os
import pandas as pd
try:
    import pandera.pandas as pa
    from pandera import Column, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

TARGET = "Life expectancy "

def get_life_expectancy_schema():
    if not PANDERA_AVAILABLE:
        return None
    return pa.DataFrameSchema({
        "Country": Column(pa.String, nullable=False), "Year": Column(pa.Int, Check.in_range(2000, 2100)),
        "Status": Column(pa.String, Check.isin(["Developed", "Developing"])),
        TARGET: Column(pa.Float, Check.in_range(0, 100), nullable=True),
        "Adult Mortality": Column(pa.Float, Check.ge(0), nullable=True), "infant deaths": Column(pa.Int, Check.ge(0)),
        "Alcohol": Column(pa.Float, Check.ge(0), nullable=True), "percentage expenditure": Column(pa.Float, Check.ge(0), nullable=True),
        "Hepatitis B": Column(pa.Float, Check.in_range(0, 100), nullable=True), "Measles ": Column(pa.Int, Check.ge(0)),
        " BMI ": Column(pa.Float, Check.ge(0), nullable=True), "under-five deaths ": Column(pa.Int, Check.ge(0)),
        "Polio": Column(pa.Float, Check.in_range(0, 100), nullable=True), "Total expenditure": Column(pa.Float, Check.ge(0), nullable=True),
        "Diphtheria ": Column(pa.Float, Check.in_range(0, 100), nullable=True), " HIV/AIDS": Column(pa.Float, Check.ge(0), nullable=True),
        "GDP": Column(pa.Float, Check.ge(0), nullable=True), "Population": Column(pa.Float, Check.ge(0), nullable=True),
        " thinness  1-19 years": Column(pa.Float, Check.ge(0), nullable=True), " thinness 5-9 years": Column(pa.Float, Check.ge(0), nullable=True),
        "Income composition of resources": Column(pa.Float, Check.in_range(0, 1), nullable=True), "Schooling": Column(pa.Float, Check.ge(0), nullable=True),
    }, strict=True)

def validate_schema(df, output_report_name="baseline_validation.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    if PANDERA_AVAILABLE:
        try:
            get_life_expectancy_schema().validate(df, lazy=True)
            print("[SUCCESS] Pandera schema validation PASSED.")
            return True
        except pa.errors.SchemaErrors as err:
            failures=err.failure_cases[['schema_context','column','check','failure_case','index']]
            os.makedirs('artifacts', exist_ok=True); path=os.path.join('artifacts',output_report_name)
            failures.to_csv(path,index=False); print(failures.to_string()); print(f"[ERROR] Schema Validation FAILED. Report: {path}"); return False
    # Lightweight fallback so the project can be checked even before dependencies are installed.
    expected=["Country","Year","Status",TARGET,"Adult Mortality","infant deaths","Alcohol","percentage expenditure","Hepatitis B","Measles "," BMI ","under-five deaths ","Polio","Total expenditure","Diphtheria "," HIV/AIDS","GDP","Population"," thinness  1-19 years"," thinness 5-9 years","Income composition of resources","Schooling"]
    missing=[c for c in expected if c not in df.columns]; extra=[c for c in df.columns if c not in expected]
    problems=[]
    if missing: problems.append(f"Missing columns: {missing}")
    if extra: problems.append(f"Unexpected columns: {extra}")
    if not df['Year'].between(2000,2100).all(): problems.append('Year contains values outside 2000-2100.')
    if not df['Status'].dropna().isin(['Developed','Developing']).all(): problems.append('Status contains unexpected values.')
    if not df[TARGET].dropna().between(0,100).all(): problems.append('Life expectancy contains invalid values.')
    if problems:
        os.makedirs('artifacts',exist_ok=True); open('artifacts/'+output_report_name,'w',encoding='utf-8').write('\n'.join(problems)); print('[ERROR] Lightweight schema validation FAILED.'); print('\n'.join(problems)); return False
    print('[SUCCESS] Lightweight schema validation PASSED. Install requirements for full Pandera validation.'); return True

if __name__ == '__main__':
    path='data/raw/Life Expectancy Data.csv'
    if os.path.exists(path):
        ok=validate_schema(pd.read_csv(path), 'baseline_validation.csv')
        raise SystemExit(0 if ok else 1)
    print(f'[ERROR] Target file not found: {path}')
    raise SystemExit(1)
