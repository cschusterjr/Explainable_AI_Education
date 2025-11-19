
import numpy as np, pandas as pd, joblib, pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

OUT = pathlib.Path("artifacts"); OUT.mkdir(exist_ok=True, parents=True)

# Synthetic "education" dataset with protected attribute proxy (e.g., group)
rng = np.random.default_rng(13)
n = 1500
X = pd.DataFrame({
    "hours_studied": rng.normal(5, 2, n).clip(0),
    "attendance": rng.normal(0.9, 0.05, n).clip(0,1),
    "prior_gpa": rng.normal(2.8, 0.6, n).clip(0,4),
    "group": rng.integers(0, 2, n),   # proxy for fairness analysis
})
y = (0.5*X["hours_studied"] + 0.8*X["attendance"] + 0.7*X["prior_gpa"] + rng.normal(0,0.4,n) > 2.2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13, stratify=y)
clf = GradientBoostingClassifier().fit(X_train.drop(columns=["group"]), y_train)
pred = clf.predict(X_test.drop(columns=["group"]))

report = classification_report(y_test, pred, output_dict=True)
pd.DataFrame(report).to_csv(OUT/"metrics.csv")

joblib.dump(clf, OUT/"model.pkl")
X_test.to_csv(OUT/"X_test.csv", index=False)
pd.Series(y_test).to_csv(OUT/"y_test.csv", index=False)
print("Artifacts saved to artifacts/")
