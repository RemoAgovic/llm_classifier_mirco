#%% MERGE – run this in a separate script or notebook cell

import os
import pandas as pd
from IPython.display import display  # for nice tables in Jupyter/VS Code

# ---------------------------
# CONFIG FOR MERGE
# ---------------------------
# 1) Original CSV (same as in your classifier config)
INPUT_CSV = r"C:\Users\RemoAgovic\OneDrive - Wyss Academy for Nature\Basis\9_GitHub\classifier\data\input_data\Test_pressures_dataframe.csv"

# 2) Folder with the FINAL merged model output for ONE run
#    -> This is the run_final_results_folder printed by the classifier script,
#       e.g.: ...\data\output_data\test\run_20250208_123456
RUN_FINAL_RESULTS_FOLDER = r"C:\Users\RemoAgovic\OneDrive - Wyss Academy for Nature\Basis\9_GitHub\classifier\data\output_data\test\run_20251208_153025"

# 3) Name for the merged file you want to create
OUTPUT_MERGED_FILENAME = "Test_pressures_dataframe_with_model_labels.csv"


# ---------------------------
# MERGE L
# ---------------------------

# 1) Paths based on your config above
original_path = INPUT_CSV
classified_path = os.path.join(RUN_FINAL_RESULTS_FOLDER, "flattened_results_all.csv")

output_merged_path = os.path.join(
    RUN_FINAL_RESULTS_FOLDER,
    OUTPUT_MERGED_FILENAME
)

print("Original CSV:", original_path)
print("Classified CSV:", classified_path)
print("Merged will be saved to:", output_merged_path)

# 2) Load data
df_orig = pd.read_csv(original_path, encoding="utf-8-sig")
df_cls = pd.read_csv(classified_path, encoding="utf-8-sig")

# 3) Recover original row id from the model's id field ("1_1" -> 1)
df_cls["orig_id"] = df_cls["id"].astype(str).str.split("_").str[0].astype(int)

# 4) Rename model columns so they don't overwrite your originals
df_cls = df_cls.rename(columns={
    "pressure": "pressure_model",
    "theme": "theme_model",
    "sentiment": "sentiment_model"
})

# Keep only the columns we need from the classified df
df_cls_small = df_cls[["orig_id", "pressure_model", "theme_model", "sentiment_model"]]

# 5) Merge: original data + model outputs
df_merged = df_orig.merge(
    df_cls_small,
    left_on="id",      # original id column
    right_on="orig_id",
    how="left"
).drop(columns=["orig_id"])

# 6) Save merged file
os.makedirs(os.path.dirname(output_merged_path), exist_ok=True)
df_merged.to_csv(output_merged_path, index=False, encoding="utf-8-sig")

print("\nMerged file saved to:")
print(output_merged_path)

# 7) Show and keep df_merged in the Jupyter/Interactive environment
display(df_merged.head())  # quick preview

df_merged  # last expression → appears as the cell output AND stays as a variable

# %%
