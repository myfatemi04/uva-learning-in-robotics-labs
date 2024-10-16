import matplotlib.pyplot as plt
import json

rows = []
with open("reward.jsonl") as f:
    for line in f.read().split("\n"):
        if line:
            rows.append(json.loads(line))
        else:
            break

rewards = [row['reward'] for row in rows]


plt.plot(rewards)
plt.savefig("learning_curve.png")