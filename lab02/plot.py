import matplotlib.pyplot as plt
import json

n=7
rows = []
with open(f"runs/{n}/reward.jsonl") as f:
    for line in f.read().split("\n"):
        if line:
            rows.append(json.loads(line))
        else:
            break

rewards = [row['reward'] for row in rows]


plt.plot(rewards)
plt.savefig(f"runs/{n}/learning_curve.png")