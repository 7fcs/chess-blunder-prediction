import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

df = pd.read_csv("moves.csv")

df["log_clock"] = np.log(df["clock_before"] + 1)
df["clock_frac"] = df["clock_before"] / df["initial_time"]
df["abs_eval"] = df["eval_before"].clip(-1000, 1000).abs()

features = [
    "player_rating",
    "rating_diff",
    "log_clock",
    "clock_frac",
    "increment",
    "move_number",
    "abs_eval",
    "num_legal_moves",
    "material_total",
    "in_check",
]

first_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
train_idx, rest_idx = next(first_split.split(df, groups=df["game_id"]))
train = df.iloc[train_idx]
rest = df.iloc[rest_idx]

second_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
val_idx, test_idx = next(second_split.split(rest, groups=rest["game_id"]))
val = rest.iloc[val_idx]
test = rest.iloc[test_idx]

splits = pd.concat([
    pd.DataFrame({"game_id": train["game_id"].unique(), "split": "train"}),
    pd.DataFrame({"game_id": val["game_id"].unique(), "split": "val"}),
    pd.DataFrame({"game_id": test["game_id"].unique(), "split": "test"}),
])
splits.to_csv("splits.csv", index=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_val = scaler.transform(val[features])
X_test = scaler.transform(test[features])
y_train = train["is_blunder"]
y_val = val["is_blunder"]
y_test = test["is_blunder"]

best_model = None
best_loss = None
for c in [0.001, 0.01, 0.1, 1, 10]:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
    print(f"C={c}  val log loss={loss:.4f}")
    if best_loss is None or loss < best_loss:
        best_loss = loss
        best_model = model

p_test = best_model.predict_proba(X_test)[:, 1]
p_base = np.full(len(y_test), y_train.mean())

print(f"\nchosen C={best_model.C}")
print(f"test AUC        {roc_auc_score(y_test, p_test):.4f}")
print(f"test log loss   {log_loss(y_test, p_test):.4f}")
print(f"base log loss   {log_loss(y_test, p_base):.4f}")
print(f"test Brier      {brier_score_loss(y_test, p_test):.4f}")

coefs = pd.Series(best_model.coef_[0], index=features)
print("\ncoefficients")
print(coefs.sort_values(key=abs, ascending=False))

frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
plt.plot(mean_pred, frac_pos, marker="o", label="logistic regression")
plt.plot([0, mean_pred.max()], [0, mean_pred.max()], linestyle="--", label="perfect")
plt.xlabel("predicted blunder probability")
plt.ylabel("actual blunder rate")
plt.legend()
plt.savefig("baseline_calibration.png")
