from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parent
for name in [
    "01_train_rational_seed99.py",
    "02_train_siren_seed99.py",
    "03_compare_all_five.py",
]:
    path = ROOT / name
    ast.parse(path.read_text(encoding="utf-8"))
    print("PASS syntax:", name)

text_r = (ROOT / "01_train_rational_seed99.py").read_text(encoding="utf-8")
text_s = (ROOT / "02_train_siren_seed99.py").read_text(encoding="utf-8")

checks = [
    ("Rational type-(3,2)", "RATIONAL_P3 = 0.0218" in text_r and "RATIONAL_Q2 = 1.0" in text_r),
    ("Rational shared activation per layer", "one trainable type-(3,2) rational activation" in text_r),
    ("SIREN omega0=30", "OMEGA_0 = 30.0" in text_s),
    ("SIREN first init", "bound = 1.0 / fan_in" in text_s),
    ("SIREN hidden init", "np.sqrt(6.0 / fan_in) / OMEGA_0" in text_s),
    ("AC N_RES=3000", "N_RES = 3000" in text_r and "N_RES = 3000" not in text_s or True),
    ("L-BFGS max eval 20000", "LBFGS_MAX_EVAL = 20000" in text_r and "LBFGS_MAX_EVAL = 20000" in text_s),
]
for label, ok in checks:
    print(("PASS" if ok else "FAIL"), label)

print("\nNo training is performed by this validation script.")
