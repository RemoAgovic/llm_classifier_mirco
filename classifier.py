# classifier.py
# Task-specific setup:
# - paths and CSV column names
# - prompt file
# - how to build the user content
# - how to send the request
# The heavy lifting is done in helper.py.

#%% CONFIG

import os
import json
from datetime import datetime
import pandas as pd
import openai
import helper

# ---------------------------
# PATHS
# ---------------------------

## --> ADJUST YOUR MAIN PATH HERE <--
MAIN = "C:\\Users\\RemoAgovic\\OneDrive - Wyss Academy for Nature\\Basis\\9_GitHub\\classifier" ## Adjust this to your main project folder 

# CSV with the sentences you want to classify
INPUT_CSV = os.path.join(MAIN, "data", "input_data", "Test_pressures_dataframe.csv")

# Root folder where you want all run folders to live
OUTPUT_ROOT = os.path.join(MAIN, "data", "output_data", "test")

# Path to prompt JSON (SYSTEM_MESSAGE + TASK_INSTRUCTIONS)
PROMPT_PATH = os.path.join(MAIN, "prompt", "prompt.json")

# ---------------------------
# OPENAI SETTINGS
# ---------------------------

MODEL_NAME = "gpt-5-mini"
#TEMPERATURE = 0.2 # only for 4o
#MAX_TOKENS = 16000   # only for 4o

# ---------------------------
# DATA COLUMN SETTINGS
# ---------------------------

# Adjust these so the wrapper knows how to read your CSV.
GROUP_COL = "id"           # which column identifies an "article" (group)? For your test, each row = its own article, so just "id".
SENTENCE_COL = "sentence"  # column with the text to classify
SENTENCE_INDEX_COL = "id"  # used to build a unique sentence ID; for your test, "id" is fine
ARTICLE_BODY_COL = "sentence"  # what to use as "article body" at the top of the prompt; for simple tests, reuse "sentence"

INCLUDE_ARTICLE_BODY = False     # include article body before the instructions
ARTICLE_BODY_MAX_CHARS = 200    # 0 = no truncation; 200 mimics your earlier script

# ---------------------------
# BATCHING / PARALLELISM
# ---------------------------

CHUNK_SIZE = 100     # THis is just to define how much at once gets sent to the API
MAX_WORKERS = 20     # This is how many parallel requests get sent to OpenAI

# ---------------------------
# HOW TO FLATTEN THE JSON
# ---------------------------
# ---> ADJUST: Here you must define your columns as you did it in the prompt <---
#
# The model must return a JSON array, one object per sentence.
# MAPPING: model_json_key -> column_name_in_output_csv

MAPPING = {
    "id": "id",
    "pressure": "pressure",
    "theme": "theme",
    "sentiment": "sentiment"
}


## This here just loads the prompt and pushes all config into the helper module.
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

SYSTEM_MESSAGE = prompt_data["SYSTEM_MESSAGE"].strip()
TASK_INSTRUCTIONS = prompt_data["TASK_INSTRUCTIONS"].strip()

# push config into helper
helper.INCLUDE_ARTICLE_BODY   = INCLUDE_ARTICLE_BODY
helper.ARTICLE_BODY_MAX_CHARS = ARTICLE_BODY_MAX_CHARS
helper.TASK_INSTRUCTIONS      = TASK_INSTRUCTIONS
helper.MODEL_NAME             = MODEL_NAME
helper.SYSTEM_MESSAGE         = SYSTEM_MESSAGE
#helper.MAX_TOKENS             = MAX_TOKENS
#helper.TEMPERATURE            = TEMPERATURE
helper.GROUP_COL              = GROUP_COL
helper.SENTENCE_COL           = SENTENCE_COL
helper.SENTENCE_INDEX_COL     = SENTENCE_INDEX_COL
helper.ARTICLE_BODY_COL       = ARTICLE_BODY_COL
helper.CHUNK_SIZE             = CHUNK_SIZE
helper.MAX_WORKERS            = MAX_WORKERS
helper.MAPPING                = MAPPING


#%% NEW RUN

# 1) Read your input CSV
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

# 2) Set up unique run folders
unique_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_data_folder = os.path.join(OUTPUT_ROOT)   # you can also create subfolders here if you like
final_data_folder = os.path.join(OUTPUT_ROOT)
os.makedirs(output_data_folder, exist_ok=True)
os.makedirs(final_data_folder, exist_ok=True)

run_partial_results_folder = os.path.join(output_data_folder, f"run_{unique_run_id}")
run_final_results_folder = os.path.join(final_data_folder, f"run_{unique_run_id}")
os.makedirs(run_partial_results_folder, exist_ok=True)
os.makedirs(run_final_results_folder, exist_ok=True)

checkpoint_path = os.path.join(run_partial_results_folder, "checkpoint.txt")

print(f"Starting new run. Partial results folder: {run_partial_results_folder}")

helper.run_classification_in_chunks(df, run_partial_results_folder, checkpoint_path)
helper.merge_partial_results_into_final(run_partial_results_folder, run_final_results_folder)

