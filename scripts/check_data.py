# check_data.py — Data quality assertions for tempe_nppes_demo.csv
# Run: python scripts/check_data.py
# Validates: required columns present, no null NPIs, all rows are AZ,
# no duplicate NPI+address pairs, no unverified "free" affordability claims.

from pathlib import Path
import pandas as pd
p=Path(__file__).parents[1]/"data/tempe_nppes_demo.csv"; df=pd.read_csv(p,dtype=str)
required={"npi","name","category","specialty","address","city","state","zip","source","affordability"}
assert not(required-set(df.columns)); assert df.npi.notna().all(); assert (df.state=="AZ").all(); assert df.duplicated(["npi","address"]).sum()==0
assert not df.affordability.str.lower().isin(["free","low cost"]).any()
print(f"OK: {len(df)} rows")
