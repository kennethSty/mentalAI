import ast
import numpy as np

files = [
    "eval/llama2"
    "eval/llama2_sentiment+suicide"
    "eval/llama2_rag1+sentiment+suicide"
]

for filename in files:
    print(f"Reading {filename}")

    values = []
    with open(filename) as file:
        for line in file:
            if line.startswith("Scores: "):
                s = line[line.find("["):line.find("]") + 1]
                values.append(ast.literal_eval(s))

    vals = np.asarray(values).flatten()

    print(f"Number of scores {vals.shape}")
    print(vals)
    print(f"Avg.: {vals.mean()}")