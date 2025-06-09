import json
import os
import random
import time
from datetime import datetime

from openai import OpenAI, OpenAIError

import sys

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------
API_KEY = os.getenv("ALI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-235b-a22b"

# API_KEY = os.getenv("SIL_API_KEY") 
# BASE_URL = "https://api.siliconflow.cn/v1"
# MODEL = "qwen3-235b-a22b"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ---------------------------------------------------------------------
# Taxonomy definition
# ---------------------------------------------------------------------
TAXONOMY = {
    "non-astronomy": {
        "math": {},
        "biology": {},
        "computer-science": {},
        "quantum-physics": {}
    },
    "astronomy": {
        "non-star-formation": {},
        "star-formation": {
            "astrochemistry": {},
            "outflow": {},
            "protostellar-disk": {},
            "high-mass-sf": {},
            "low-mass-sf": {},
            "prestellar-core": {},
            "filamentary-cloud": {},
            "protocluster": {},
            "fragmentation": {},
            "hub-filament-system": {}
        }
    }
}

# ---------------------------------------------------------------------
# Target number of samples per leaf
# ---------------------------------------------------------------------
TARGET_NUM = {
    "non-star-formation": 5,
    "math": 5,
    "biology": 5,
    "computer-science": 5,
    "quantum-physics": 5,
}
DEFAULT_SAMPLES = 5

# ---------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------
SYSTEM_TEMPLATE = (
    "You are an experienced researcher in {domain}, writing concise, publish-quality scientific "
    "abstracts in English."
)

USER_TEMPLATE = (
    "Generate {n} paper entries strictly on \"{topic}\".\n"
    "Return a JSON array; each element must contain:\n"
    '{{"title": "<max 20 words>", '
    '"abstract": "<150-250 words scientific abstract>", '
    '"keywords": "<3-6 comma-separated keywords>"}}\n'
    "The abstract must be original and structured like a scientific paper "
    "(Background, Methods, Results, Conclusions). "
    "If no relevant papers can be generated, return an empty JSON array []. "
    "Output JSON only."
)

# ---------------------------------------------------------------------
# Utility: iterate leaf nodes
# ---------------------------------------------------------------------
def leaf_paths(tree, prefix=()):
    for key, subtree in tree.items():
        if subtree:
            yield from leaf_paths(subtree, prefix + (key,))
        else:
            yield prefix + (key,)

# ---------------------------------------------------------------------
# Generation function for a single leaf
# ---------------------------------------------------------------------
def generate_batch(path, n, max_retry=2):
    if path[0] == 'non-astronomy':
        path = path[1:]  
    domain = path[0]
    topic = "/".join(path)
    # print(domain)
    # print(topic)
    # # sys.exit(0)
    system_msg = SYSTEM_TEMPLATE.format(domain=domain)
    user_msg = USER_TEMPLATE.format(n=n, topic=topic)

    temperature = round(random.uniform(0.85, 1.0), 2)

    for attempt in range(max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                extra_body={"enable_thinking": False}
            )
            raw_content = resp.choices[0].message.content

            if not raw_content or raw_content.strip() == "":
                raise RuntimeError("Empty response from model.")

            try:
                data = json.loads(raw_content)
                print(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for topic: {topic}")
                print(f"Raw content was: {repr(raw_content)}")
                raise e

            break  # success
        except (OpenAIError, RuntimeError, json.JSONDecodeError) as e:
            print(f"Retry {attempt+1}/{max_retry} for leaf {'/'.join(path)}... Reason: {e}")
            time.sleep(2)
            continue
    else:
        # All retries failed
        raise RuntimeError(f"Failed after {max_retry+1} attempts: {topic}")

    for entry in data:
        entry["labels"] = list(path)
    return data

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main(output_file="synthetic_corpus.jsonl"):
    with open(output_file, "w", encoding="utf-8") as f:
        for path in leaf_paths(TAXONOMY):
            leaf = path[-1]
            n_samples = TARGET_NUM.get(leaf, DEFAULT_SAMPLES)
            print(f"➡️  Generating {n_samples:3d} samples for {'/'.join(path)}")

            try:
                batch = generate_batch(path, n_samples)
            except RuntimeError as e:
                print(f"Failed to generate for {'/'.join(path)}: {e}")
                continue  # Skip this leaf and continue

            for rec in batch:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # Flush to disk immediately
            print(f"Saved {len(batch)} samples for {'/'.join(path)}")

            time.sleep(1)

    print(f"🎉 All done. Output written to {output_file}")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if os.getenv("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY_HERE") == "YOUR_DASHSCOPE_API_KEY_HERE":
        raise SystemExit("Please set your DASHSCOPE_API_KEY environment variable or edit the script.")
    main()
