# Text Classifier

Batch-classify sentences/articles with OpenAI (e.g. GPT-5 mini / nano).  
You provide a CSV and a prompt; the scripts handle chunking, calls, and saving results.

---

## 1. What you need to provide

### a) Input CSV

Put your CSV here:

`data/input_data/your_data.csv`

Default expectation:

- `id` – unique ID per row
- `sentence` – text to classify

You can change these names in `classifier.py` (see below).

### b) Prompt

Edit this file:

`prompt/prompt.json`

It must contain:

```json
{
  "SYSTEM_MESSAGE": "…",
  "TASK_INSTRUCTIONS": "…"
}

Make sure to also adapt the items to be classified in the script. They must be names identically in the json instructions and in the script!

classify.py is the script you need to run the classification.

merge.py is the script you need if you want to append the classification back to your original data and do a quick check. 
