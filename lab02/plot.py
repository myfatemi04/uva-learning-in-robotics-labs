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

names = ["reward", "policy", "vf", "entropy"]
for i in range(4):
    values = [row[names[i]] for row in rows]
    plt.subplot(2, 2, i + 1)
    plt.plot(values)
    plt.xlabel("Timestep")
    plt.ylabel(names[i].title())
    if names[i] == 'reward':
        plt.ylim(0, max(values) * 1.1)

plt.tight_layout()
plt.savefig(f"runs/{n}/learning_curve.png")
