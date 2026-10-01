import json
import statistics

data = json.load(open("scripts/split_experiment_results.json"))
runs3 = {r["seed"]: r for r in data["by_num_indices"]["3"]["runs"] if "error" not in r}
runs4 = {r["seed"]: r for r in data["by_num_indices"]["4"]["runs"] if "error" not in r}

common = sorted(set(runs3) & set(runs4))
diffs = []          # test_rmse(4) - test_rmse(3); negative means 4 is better
wins4 = 0
for s in common:
    t3 = runs3[s]["test"]["rmse"]
    t4 = runs4[s]["test"]["rmse"]
    d = t4 - t3
    diffs.append(d)
    if d < 0:
        wins4 += 1

n = len(common)
print(f"paired runs: {n}")
print(f"index-4 beat index-3 on test RMSE: {wins4}/{n} = {100*wins4/n:.1f}% of splits")
print(f"mean(test4 - test3): {statistics.mean(diffs):+.5f}  (negative = 4 better)")
print(f"std of the difference: {statistics.pstdev(diffs):.5f}")
# a difference much smaller than its std => no reliable winner