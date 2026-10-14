from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


# ============================================================
# Paths
# ============================================================

DIST_PATH = Path(
    "generation/analysis/"
    "visual_distinctiveness/"
    "item_visual_distinctiveness.csv"
)

GENRE_PATH = Path(
    "generation/analysis/"
    "genre_multiseed/"
    "item_genre_corrected_multiseed_gate.csv"
)

OUTPUT_DIR = Path(
    "generation/analysis/"
    "genre_controlled_regression"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OLS helper
# ============================================================

def fit_ols(X, y):

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0]

    y_pred = X @ beta
    residuals = y - y_pred

    n = len(y)
    p = X.shape[1]

    sse = np.sum(
        residuals ** 2
    )

    sst = np.sum(
        (y - y.mean()) ** 2
    )

    r2 = 1.0 - sse / sst

    adjusted_r2 = (
        1.0 -
        (1.0 - r2) *
        (n - 1) /
        (n - p)
    )

    # Conventional OLS variance estimate
    sigma2 = (
        sse /
        (n - p)
    )

    xtx_inv = np.linalg.pinv(
        X.T @ X
    )

    var_beta = (
        sigma2 *
        xtx_inv
    )

    se = np.sqrt(
        np.diag(var_beta)
    )

    t_values = (
        beta /
        se
    )

    p_values = (
        2.0 *
        t.sf(
            np.abs(t_values),
            df=n - p
        )
    )

    return {
        "beta": beta,
        "se": se,
        "t": t_values,
        "p": p_values,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "y_pred": y_pred,
        "residuals": residuals,
    }


# ============================================================
# 1. Load data
# ============================================================

print("=" * 75)
print("Genre-Controlled Visual Distinctiveness Regression")
print("=" * 75)


dist_df = pd.read_csv(
    DIST_PATH
)

genre_df = pd.read_csv(
    GENRE_PATH
)


genre_df = (
    genre_df[
        ["item_id", "genre"]
    ]
    .drop_duplicates()
)


print(
    "\nItems in distinctiveness data:",
    dist_df["item_id"].nunique()
)

print(
    "Genre-tagged items:",
    genre_df["item_id"].nunique()
)


# ============================================================
# 2. Build multi-hot genre matrix
# ============================================================

genre_matrix = pd.crosstab(
    genre_df["item_id"],
    genre_df["genre"]
)

genre_matrix = (
    genre_matrix > 0
).astype(float)

print(
    "\nGenre matrix shape:",
    genre_matrix.shape
)

print(
    "Genres:"
)

for genre in genre_matrix.columns:
    print("-", genre)


# ============================================================
# 3. Merge
# ============================================================

data = (
    dist_df[
        [
            "item_id",
            "image_weight_mean",
            "visual_distinctiveness",
        ]
    ]
    .merge(
        genre_matrix,
        left_on="item_id",
        right_index=True,
        how="inner"
    )
)

print(
    "\nRegression items:",
    len(data)
)

assert len(data) == 7425


# ============================================================
# 4. Standardize continuous variables
# ============================================================

y_raw = data[
    "image_weight_mean"
].to_numpy(
    dtype=np.float64
)

d_raw = data[
    "visual_distinctiveness"
].to_numpy(
    dtype=np.float64
)


y = (
    y_raw - y_raw.mean()
) / y_raw.std(
    ddof=0
)

d = (
    d_raw - d_raw.mean()
) / d_raw.std(
    ddof=0
)


genre_columns = list(
    genre_matrix.columns
)

G = data[
    genre_columns
].to_numpy(
    dtype=np.float64
)


# ============================================================
# 5. Model A:
#    Genre only
# ============================================================

intercept = np.ones(
    (len(data), 1),
    dtype=np.float64
)

X_genre = np.column_stack([
    intercept,
    G,
])


genre_model = fit_ols(
    X_genre,
    y
)


print("\n" + "=" * 75)
print("MODEL A: GENRE ONLY")
print("=" * 75)

print(
    f"R²          = "
    f"{genre_model['r2']:.6f}"
)

print(
    f"Adjusted R² = "
    f"{genre_model['adjusted_r2']:.6f}"
)


# ============================================================
# 6. Model B:
#    Genre + visual distinctiveness
# ============================================================

X_full = np.column_stack([
    intercept,
    d,
    G,
])


full_model = fit_ols(
    X_full,
    y
)


print("\n" + "=" * 75)
print("MODEL B: GENRE + VISUAL DISTINCTIVENESS")
print("=" * 75)

print(
    f"R²          = "
    f"{full_model['r2']:.6f}"
)

print(
    f"Adjusted R² = "
    f"{full_model['adjusted_r2']:.6f}"
)


# ============================================================
# 7. Distinctiveness coefficient
# ============================================================

# Column:
# 0 = intercept
# 1 = standardized distinctiveness
# 2 onward = genre indicators

beta_dist = (
    full_model["beta"][1]
)

se_dist = (
    full_model["se"][1]
)

t_dist = (
    full_model["t"][1]
)

p_dist = (
    full_model["p"][1]
)


print("\n" + "=" * 75)
print("DISTINCTIVENESS AFTER CONTROLLING FOR GENRE")
print("=" * 75)

print(
    f"Standardized beta = "
    f"{beta_dist:.6f}"
)

print(
    f"SE                = "
    f"{se_dist:.6f}"
)

print(
    f"t                 = "
    f"{t_dist:.6f}"
)

print(
    f"p                 = "
    f"{p_dist:.3e}"
)


# ============================================================
# 8. Incremental R²
# ============================================================

delta_r2 = (
    full_model["r2"] -
    genre_model["r2"]
)


print("\n" + "=" * 75)
print("INCREMENTAL EXPLANATORY POWER")
print("=" * 75)

print(
    f"Genre-only R²              : "
    f"{genre_model['r2']:.6f}"
)

print(
    f"Genre + distinctiveness R² : "
    f"{full_model['r2']:.6f}"
)

print(
    f"ΔR² from distinctiveness   : "
    f"{delta_r2:.6f}"
)


# ============================================================
# 9. Save coefficient table
# ============================================================

feature_names = (
    ["Intercept", "Visual distinctiveness"]
    +
    genre_columns
)


coef_df = pd.DataFrame({
    "feature":
        feature_names,

    "beta":
        full_model["beta"],

    "std_error":
        full_model["se"],

    "t_value":
        full_model["t"],

    "p_value":
        full_model["p"],
})


coef_path = (
    OUTPUT_DIR /
    "genre_controlled_coefficients.csv"
)

coef_df.to_csv(
    coef_path,
    index=False
)


# ============================================================
# 10. Save summary
# ============================================================

summary_df = pd.DataFrame([
    {
        "model":
            "Genre only",

        "r2":
            genre_model["r2"],

        "adjusted_r2":
            genre_model["adjusted_r2"],
    },

    {
        "model":
            "Genre + visual distinctiveness",

        "r2":
            full_model["r2"],

        "adjusted_r2":
            full_model["adjusted_r2"],
    },
])


summary_path = (
    OUTPUT_DIR /
    "regression_model_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("\nSaved:")
print(coef_path)
print(summary_path)

print(
    "\nGenre-controlled regression completed."
)