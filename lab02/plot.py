import matplotlib.pyplot as plt
import json,sys
n=sys.argv[1]
rows = []
with open(f"runs/{n}/reward.jsonl") as f:
    for line in f.read().split("\n"):
        if line:
            rows.append(json.loads(line))
        else:
            break

rewards = [row['reward'] for row in rows]

plt.plot(rewards)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.savefig(f"runs/{n}/learning_curve.png")