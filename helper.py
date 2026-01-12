# helper.py
# Generic engine: chunking, parallel requests, flattening, merging.
# All task-specific things (prompts, model, etc.) live in classifier.py.

import os
import time
import json
import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor


# NOTE:

# These are set from classifier.py:
#   INCLUDE_ARTICLE_BODY, TASK_INSTRUCTIONS
#   MODEL_NAME, SYSTEM_MESSAGE, MAX_TOKENS, TEMPERATURE
#   GROUP_COL, SENTENCE_COL, SENTENCE_INDEX_COL, ARTICLE_BODY_COL
#   ARTICLE_BODY_MAX_CHARS, CHUNK_SIZE, MAX_WORKERS, MAPPING



## Here you can adjust if there should for example be articile body + prompt + sentence or swap the order etc.
def build_user_content(article_group):
    """
    Build the user content for the OpenAI API.
    Uses:
    - INCLUDE_ARTICLE_BODY
    - TASK_INSTRUCTIONS
    - the article structure that helper.run_classification_in_chunks builds.
    """
    user_content_lines = []

    # Add article body (only first article in the group; usually groups of size 1)
    for article in article_group:
        if INCLUDE_ARTICLE_BODY:
            user_content_lines.append(f"Article ID: {article['id']}")
            user_content_lines.append("Article Body:")
            user_content_lines.append(article.get("article_body", "") or "")
            user_content_lines.append("")

    # Add task instructions
    user_content_lines.append(TASK_INSTRUCTIONS)
    user_content_lines.append("")

    # Add sentence list
    for article in article_group:
        user_content_lines.append(f"Article ID: {article['id']}")
        for s in article["sentences"]:
            user_content_lines.append(f"{s['id_article_sent']}. \"{s['text']}\"")
        user_content_lines.append("")

    return "\n".join(user_content_lines)


def send_request(request_id, article_group):
    """
    Send one request to the OpenAI API for a single article_group.
    Uses:
    - MODEL_NAME, SYSTEM_MESSAGE, MAX_TOKENS, TEMPERATURE
    - build_user_content(...) above.
    """
    user_content = build_user_content(article_group)

    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": user_content}
        ],
        #"max_tokens": MAX_TOKENS,
        #"temperature": TEMPERATURE,
    }

    try:
        response = openai.chat.completions.create(**body)
        print(f"Request {request_id} completed successfully.")
        return {
            "request_id": request_id,
            "response": response,
            "status": "success"
        }
    except Exception as e:
        print(f"Request {request_id} failed with error: {e}")
        return {
            "request_id": request_id,
            "response": None,
            "status": "failed",
            "error": str(e)
        }


def process_requests_in_parallel(grouped_articles, request_fn, max_workers):
    """
    Run many requests in parallel.

    grouped_articles: list of [article_dict] (each element is a list of length 1)
    request_fn: function(request_id, article_group) -> result dict
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(request_fn, request_id, article_group)
            for request_id, article_group in enumerate(grouped_articles)
        ]

        for i, future in enumerate(futures):
            r = future.result()
            # attach article_id for convenience
            r["article_id"] = grouped_articles[i][0]["id"]
            results.append(r)

    return results


def flatten_results(batch_results, mapping=None):
    """
    Flatten model responses into a DataFrame.

    Expects each successful result to have a 'response' with:
        response["choices"][0]["message"]["content"]
    containing a JSON array (list of objects).

    mapping: dict model_key -> output_column_name
             If None, we store the raw JSON per row.
    """
    flattened_data = []
    failed_data = []

    for result in batch_results:
        if result["status"] == "success" and result["response"] is not None:
            try:
                # response may be an OpenAI object OR already a plain dict
                resp = result["response"]
                if hasattr(resp, "to_dict"):
                    response_dict = resp.to_dict()
                else:
                    response_dict = resp  # already a dict

                response_content = response_dict["choices"][0]["message"]["content"]
                json_data = json.loads(response_content)

                if isinstance(json_data, list):
                    for entry in json_data:
                        row = {
                            "request_id": result["request_id"],
                            "article_id": result.get("article_id", None),
                        }

                        if mapping is not None:
                            for json_key, col_name in mapping.items():
                                row[col_name] = entry.get(json_key, None)
                        else:
                            # generic fallback: keep full JSON as string
                            row["raw_json"] = json.dumps(entry, ensure_ascii=False)

                        flattened_data.append(row)
                else:
                    failed_data.append({
                        "request_id": result["request_id"],
                        "article_id": result.get("article_id", None),
                        "error": "Unexpected JSON format (not a list)"
                    })

            except Exception as e:
                failed_data.append({
                    "request_id": result["request_id"],
                    "article_id": result.get("article_id", None),
                    "error": str(e)
                })
        else:
            failed_data.append({
                "request_id": result["request_id"],
                "article_id": result.get("article_id", None),
                "error": result.get("error", "No response object")
            })

    flattened_df = pd.DataFrame(flattened_data)
    failed_df = pd.DataFrame(failed_data)

    return flattened_df, failed_df


def run_classification_in_chunks(df, run_partial_results_folder, checkpoint_path):
    """
    Main loop:
    - builds article structures
    - sends requests in chunks
    - saves partial JSON + CSV
    - supports resume via checkpoint file
    """
    print("Starting classification in chunks...")

    overall_start_time = time.time()

    # Derived sentence ID like in your original script
    df["id_article_sent"] = df[GROUP_COL].astype(str) + "_" + df[SENTENCE_INDEX_COL].astype(str)

    # Optionally truncate article body
    if INCLUDE_ARTICLE_BODY and ARTICLE_BODY_MAX_CHARS > 0:
        df[ARTICLE_BODY_COL] = df[ARTICLE_BODY_COL].astype(str).str[:ARTICLE_BODY_MAX_CHARS]

    # Build article structures
    article_groups = df.groupby(GROUP_COL)
    articles = [
        {
            "id": article_id,
            "article_body": group[ARTICLE_BODY_COL].iloc[0] if ARTICLE_BODY_COL in group.columns else "",
            "sentences": group[["id_article_sent", SENTENCE_COL]]
                .rename(columns={SENTENCE_COL: "text"})
                .to_dict(orient="records")
        }
        for article_id, group in article_groups
    ]

    total_articles = len(articles)
    print(f"Total articles to process: {total_articles}")

    # Check for existing checkpoint
    last_completed_chunk = 0
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as cp:
                last_completed_chunk = int(cp.read().strip())
            print(f"Resuming from chunk {last_completed_chunk + 1}")
        except Exception:
            print("Invalid checkpoint file. Starting from the beginning.")
            last_completed_chunk = 0
    else:
        print("No checkpoint file found. Starting from the beginning.")

    cumulative_tokens = 0

    for start_idx in range(0, total_articles, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total_articles)
        chunk_index = (start_idx // CHUNK_SIZE) + 1
        articles_in_chunk = articles[start_idx:end_idx]

        partial_json_path = os.path.join(run_partial_results_folder, f"batch_results_{chunk_index}.json")
        partial_csv_path = os.path.join(run_partial_results_folder, f"flattened_results_{chunk_index}.csv")
        partial_failed_path = os.path.join(run_partial_results_folder, f"failed_results_{chunk_index}.csv")

        # Skip already processed chunks
        if os.path.exists(partial_json_path) and os.path.exists(partial_csv_path):
            print(f"Chunk {chunk_index} already processed (files exist). Skipping.")
            with open(checkpoint_path, "w", encoding="utf-8") as cp:
                cp.write(str(chunk_index))
            continue

        # Build grouped_articles: one article per request
        grouped_articles = [articles_in_chunk[i:i+1] for i in range(0, len(articles_in_chunk), 1)]

        print(f"Processing chunk {chunk_index} ({len(articles_in_chunk)} articles)...")
        batch_start_time = time.time()

        # 1) Send requests in parallel using send_request from this module
        batch_results = process_requests_in_parallel(grouped_articles, send_request, MAX_WORKERS)

        # 2) Serialize + save raw results; track tokens
        serializable_results = []
        batch_tokens = 0
        for result in batch_results:
            if result["status"] == "success" and result["response"] is not None:
                response_dict = result["response"].to_dict()
                result["response"] = response_dict
                batch_tokens += response_dict.get("usage", {}).get("total_tokens", 0)
            serializable_results.append(result)

        os.makedirs(os.path.dirname(partial_json_path), exist_ok=True)
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)

        # 3) Flatten + failures
        flattened_df, failed_df = flatten_results(serializable_results, mapping=MAPPING)

        if flattened_df.empty:
            print(f"Warning: Flattened DataFrame is empty for chunk {chunk_index}.")
        else:
            flattened_df.to_csv(partial_csv_path, index=False, encoding="utf-8-sig")

        if not failed_df.empty:
            failed_df.to_csv(partial_failed_path, index=False, encoding="utf-8-sig")
            print(f"Failed requests saved to: {partial_failed_path}")

        cumulative_tokens += batch_tokens

        batch_elapsed_time = time.time() - batch_start_time
        apm = (len(articles_in_chunk) / (batch_elapsed_time / 60)) if batch_elapsed_time > 0 else 0
        print(
            f"Finished chunk {chunk_index}. "
            f"Tokens this batch: {batch_tokens}. "
            f"Total tokens so far: {cumulative_tokens}. "
            f"Elapsed: {batch_elapsed_time:.2f}s. "
            f"Pace: {apm:.2f} articles/min."
        )

        # Update checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as cp:
            cp.write(str(chunk_index))

    overall_elapsed_time = time.time() - overall_start_time
    print(f"All chunks processed. Total elapsed time: {overall_elapsed_time:.2f} seconds.")


def merge_partial_results_into_final(run_partial_results_folder, run_final_results_folder, final_csv_name="flattened_results_all.csv"):
    """
    Merge all partial CSVs (flattened_results_*.csv) into one final CSV.
    """
    partial_csv_files = [
        f for f in os.listdir(run_partial_results_folder)
        if f.startswith("flattened_results_") and f.endswith(".csv")
    ]
    partial_csv_paths = [os.path.join(run_partial_results_folder, f) for f in partial_csv_files]

    df_list = []
    for csv_path in partial_csv_paths:
        df_chunk = pd.read_csv(csv_path, encoding="utf-8-sig")
        df_list.append(df_chunk)

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        os.makedirs(run_final_results_folder, exist_ok=True)
        final_csv_path = os.path.join(run_final_results_folder, final_csv_name)
        merged_df.to_csv(final_csv_path, index=False, encoding="utf-8-sig")
        print(f"Final merged CSV saved at: {final_csv_path}")
    else:
        print("No partial CSV files found to merge.")
