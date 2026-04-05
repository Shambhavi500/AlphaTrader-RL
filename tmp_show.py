import json

data = json.load(open('logs/agent_test.json'))
lines = []
for k, v in data.items():
    ret = v['return_pct']
    sharpe = v['sharpe']
    maxdd = v['max_dd']
    lines.append(f"{k}: return={ret:.2f}%  sharpe={sharpe:.3f}  max_dd={maxdd:.2f}%")

text = "\n".join(lines)
print(text)
open('logs/agent_summary.txt', 'w').write(text + "\n")
