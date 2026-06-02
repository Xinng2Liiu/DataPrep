import pandas as pd

mask_path = "datasets/beers/beers_dirty_error_detection.csv"

mask = pd.read_csv(mask_path)

for col in ["index", "Unnamed: 0"]:
    if col in mask.columns:
        mask = mask.drop(columns=[col])

mask = mask.replace({
    "True": True, "False": False,
    "true": True, "false": False,
    "1": True, "0": False,
    1: True, 0: False,
}).astype(bool)

print("========== beers error count by column ==========")
print(mask.sum().sort_values(ascending=False))

print("\nTotal dirty cells:", int(mask.values.sum()))
print("Total dirty rows:", int(mask.any(axis=1).sum()))

rule_cols = ["brewery_name", "city", "state"]
rule_cols = [c for c in rule_cols if c in mask.columns]

print("\n========== rule covered errors ==========")
print(mask[rule_cols].sum().sort_values(ascending=False))
print("covered dirty cells:", int(mask[rule_cols].values.sum()))
print("covered dirty rows:", int(mask[rule_cols].any(axis=1).sum()))

print("\ncell coverage:", mask[rule_cols].values.sum() / max(mask.values.sum(), 1))
print("row coverage:", mask[rule_cols].any(axis=1).sum() / max(mask.any(axis=1).sum(), 1))

print("\nounces dirty count:", int(mask["ounces"].sum()) if "ounces" in mask.columns else "no ounces")
print("ounces total rows:", len(mask))