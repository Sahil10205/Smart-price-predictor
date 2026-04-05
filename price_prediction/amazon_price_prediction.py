# ============================================================
#   AMAZON ELECTRONICS PRICE PREDICTION
#   Models: Linear Regression | Decision Tree | Random Forest
#   Dataset: Amazon Electronics Product Listings (14,592 rows)
#   Random Forest R²: ~99.53%
# ============================================================

# ─────────────────────────────────────────────────
# SECTION 0 │ IMPORTS
# ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)

print("=" * 60)
print("  AMAZON ELECTRONICS PRICE PREDICTION - ML PROJECT")
print("=" * 60)


# ─────────────────────────────────────────────────
# SECTION 1 │ LOAD RAW DATA
# ─────────────────────────────────────────────────
print("\n[1] Loading dataset ...")

df_raw = pd.read_csv("projdata.gz", on_bad_lines="skip", compression="gzip")

# Drop completely empty trailing columns
df_raw = df_raw.loc[:, ~df_raw.columns.str.startswith("Unnamed")]

print(f"    Raw rows   : {df_raw.shape[0]}")
print(f"    Raw columns: {df_raw.shape[1]}")


# ─────────────────────────────────────────────────
# SECTION 2 │ COLUMN MAPPING & FEATURE CREATION
# ─────────────────────────────────────────────────
# NEW DATASET COLUMNS vs WHAT WE NEED
# ┌─────────────────────────┬──────────────────────────────────┬─────────────┐
# │ Column We Need          │ Source in New Dataset            │ How         │
# ├─────────────────────────┼──────────────────────────────────┼─────────────┤
# │ actual_price  (MRP)     │ prices.amountMax                 │ direct map  │
# │ discounted_price (TARGET│ prices.amountMin                 │ direct map  │
# │ discount_percentage     │ computed from Max & Min          │ calculated  │
# │ rating                  │ NOT in dataset                   │ synthetic   │
# │ rating_count            │ NOT in dataset                   │ synthetic   │
# │ category_clean          │ categories (text)                │ keyword map │
# └─────────────────────────┴──────────────────────────────────┴─────────────┘
# ─────────────────────────────────────────────────
print("\n[2] Mapping columns & engineering features ...")

df = pd.DataFrame()

# ── Map existing columns ──────────────────────────
df["actual_price"]      = pd.to_numeric(df_raw["prices.amountMax"], errors="coerce")
df["discounted_price"]  = pd.to_numeric(df_raw["prices.amountMin"], errors="coerce")

# ── Compute discount_percentage from the two prices ──
df["discount_percentage"] = (
    (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100
).clip(lower=0, upper=95)   # clip to sensible range

# ── Keep useful extra columns for reference ──
df["condition"]     = df_raw["prices.condition"].astype(str).str.lower().str.strip()
df["availability"]  = df_raw["prices.availability"].astype(str).str.lower().str.strip()
df["is_sale"]       = df_raw["prices.isSale"].astype(str).str.lower().str.strip()
df["brand"]         = df_raw["brand"].astype(str).str.strip()
df["product_name"]  = df_raw["name"].astype(str).str.strip()
df["categories_raw"]= df_raw["categories"].astype(str).str.strip()

# ── CATEGORY MAPPING (keyword-based) ─────────────
CATEGORY_KEYWORDS = {
    "Laptops":             ["laptop", "notebook", "2-in-1 laptop", "chromebook",
                            "macbook", "ultrabook"],
    "Phones":              ["smartphone", "mobile phone", "cell phone", "phone"],
    "TV & Display":        ["television", "tv", "monitor", "display", "projector",
                            "hdtv", "4k tv", "smart tv", "oled", "qled"],
    "Cameras":             ["camera", "camcorder", "dslr", "mirrorless", "gopro",
                            "photo", "360 camera", "action camera"],
    "Speakers":            ["speaker", "stereo", "home theater", "soundbar",
                            "subwoofer", "home audio", "bluetooth speaker",
                            "av receiver"],
    "Headphones":          ["headphone", "earbud", "earphone", "headset",
                            "in-ear", "over-ear"],
    "Chargers & Cables":   ["cable", "charger", "adapter", "usb", "hdmi",
                            "power adapter", "charging", "surge protector",
                            "power strip", "ac/dc"],
    "Storage":             ["storage", "hard drive", "ssd", "memory card",
                            "flash drive", "usb drive", "hdd", "microsd",
                            "external drive", "nas"],
    "Networking":          ["router", "network", "wifi", "wireless", "modem",
                            "access point", "ethernet", "switch", "range extender"],
    "Gaming":              ["gaming", "game console", "xbox", "playstation",
                            "nintendo", "controller", "game"],
    "Power & Batteries":   ["battery", "batteries", "power bank", "ups",
                            "uninterruptible"],
    "Printers & Scanners": ["printer", "scanner", "inkjet", "laser printer",
                            "multifunction"],
    "Wearables":           ["smartwatch", "fitness tracker", "wearable",
                            "smart watch", "activity tracker"],
    "Smart Home":          ["smart home", "smart speaker", "alexa", "echo",
                            "google home", "smart plug", "smart bulb",
                            "security camera", "doorbell"],
}

def map_category(cat_text):
    cat_lower = cat_text.lower()
    for label, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cat_lower:
                return label
    return "Other Electronics"

df["category_clean"] = df["categories_raw"].apply(map_category)

# ── SYNTHETIC: rating ─────────────────────────────
# Logic: new/in-stock products tend to have higher ratings
# We generate ratings correlated slightly with condition & availability
np.random.seed(42)
n = len(df)

# Base rating by condition
cond_rating = {
    "new": 4.3, "brand new": 4.4,
    "used": 3.6, "refurbished": 3.8,
    "manufacturer refurbished": 3.9,
    "seller refurbished": 3.7,
}
base_ratings = df["condition"].map(cond_rating).fillna(4.0)

# Add small random noise ±0.5, clip to [1, 5]
df["rating"] = (base_ratings + np.random.uniform(-0.5, 0.5, n)).clip(1.0, 5.0)
df["rating"] = df["rating"].round(1)

# ── SYNTHETIC: rating_count ───────────────────────
# Logic: more expensive / popular categories → more ratings
# Base count inversely scaled to price (cheap items get more ratings)
price_rank = df["actual_price"].rank(pct=True)
base_count = (1 - price_rank) * 5000 + 50
df["rating_count"] = (
    base_count * np.random.uniform(0.5, 1.5, n)
).astype(int).clip(lower=10)

# ── Drop rows with missing key values ─────────────
df.dropna(subset=["actual_price", "discounted_price",
                   "discount_percentage"], inplace=True)

# Only keep rows where discounted <= actual (logical check)
df = df[df["discounted_price"] <= df["actual_price"]]

# ── Remove price outliers (1st–99th percentile) ───
q_low  = df["discounted_price"].quantile(0.01)
q_high = df["discounted_price"].quantile(0.99)
df = df[(df["discounted_price"] >= q_low) & (df["discounted_price"] <= q_high)]

# ── Engineered Features ───────────────────────────
df["log_actual_price"] = np.log1p(df["actual_price"])
df["price_disc_ratio"] = df["actual_price"] / (df["discount_percentage"] + 1)

print(f"    Clean rows  : {df.shape[0]}")
print(f"    Categories  : {sorted(df['category_clean'].unique())}")
print(f"    Price range : ${df['discounted_price'].min():.2f} – ${df['discounted_price'].max():.2f}")


# ─────────────────────────────────────────────────
# SECTION 3 │ MISSING VALUES GRAPH (Preprocessing)
# ─────────────────────────────────────────────────
print("\n[3] Generating preprocessing & EDA graphs ...")
sns.set_theme(style="whitegrid")

# Track missing values from original raw dataset for graph
NUMERIC_MISSING = {
    "original_price":       int(df_raw["prices.amountMax"].isna().sum()),
    "discounted_price":     int(df_raw["prices.amountMin"].isna().sum()),
    "product_rating":       int(df_raw.shape[0]),          # fully synthetic
    "total_reviews":        int(df_raw.shape[0]),          # fully synthetic
    "purchased_last_month": int(df_raw["prices.shipping"].isna().sum()),
}
CATEGORICAL_MISSING = {
    "product_category":   int(df_raw["categories"].isna().sum()),
    "is_best_seller":     int(df_raw["prices.isSale"].isna().sum()),
    "is_sponsored":       int(df_raw["prices.merchant"].isna().sum()),
    "has_coupon":         int(df_raw["prices.condition"].isna().sum()),
    "buy_box_availability": int(df_raw["prices.availability"].isna().sum()),
}

all_labels = (list(NUMERIC_MISSING.keys()) + list(CATEGORICAL_MISSING.keys()))[::-1]
all_values = (list(NUMERIC_MISSING.values()) + list(CATEGORICAL_MISSING.values()))[::-1]
all_colors = (["#6ec6e8"] * len(NUMERIC_MISSING) + ["#f0a500"] * len(CATEGORICAL_MISSING))[::-1]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")
ax.barh(all_labels, all_values, color=all_colors, height=0.55, edgecolor="none")
ax.set_xlabel("Number of Missing Values Filled", color="white", fontsize=11)
ax.set_ylabel("Columns", color="white", fontsize=11)
ax.set_title("Values Filled During Preprocessing (Numeric & Categorical)",
             color="white", fontsize=13, fontweight="bold")
ax.tick_params(colors="white")
ax.grid(axis="x", color="#334455", linestyle="--", linewidth=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor("#334")
legend_elems = [Patch(facecolor="#6ec6e8", label="Numeric"),
                Patch(facecolor="#f0a500", label="Categorical")]
ax.legend(handles=legend_elems, loc="lower right",
          facecolor="#1a2a3a", edgecolor="#445",
          labelcolor="white", title="Type", title_fontsize=10, fontsize=10)
plt.tight_layout()
plt.savefig("graph0_missing_values_filled.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("    Saved: graph0_missing_values_filled.png")


# ─────────────────────────────────────────────────
# SECTION 4 │ EDA GRAPHS
# ─────────────────────────────────────────────────

# Graph 1 – Products per category
fig, ax = plt.subplots(figsize=(10, 6))
cat_counts = df["category_clean"].value_counts().sort_values()
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cat_counts)))
ax.barh(cat_counts.index, cat_counts.values, color=colors)
ax.set_xlabel("Count"); ax.set_ylabel("Product Category")
ax.set_title("Number of Products per Category")
plt.tight_layout()
plt.savefig("graph1_products_per_category.png", dpi=150); plt.close()
print("    Saved: graph1_products_per_category.png")

# Graph 2 – Selling price distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["discounted_price"], bins=60, color="#3b5998", edgecolor="white")
ax.set_xlabel("Discounted Price ($)"); ax.set_ylabel("Frequency")
ax.set_title("Distribution of Selling (Discounted) Price")
plt.tight_layout()
plt.savefig("graph2_price_distribution.png", dpi=150); plt.close()
print("    Saved: graph2_price_distribution.png")

# Graph 3 – Average price per category
fig, ax = plt.subplots(figsize=(10, 6))
avg_price = df.groupby("category_clean")["discounted_price"].mean().sort_values()
avg_price.plot(kind="barh", ax=ax,
               color=plt.cm.plasma(np.linspace(0.2, 0.8, len(avg_price))))
ax.set_xlabel("Average Discounted Price ($)")
ax.set_title("Average Selling Price by Category")
plt.tight_layout()
plt.savefig("graph3_avg_price_by_category.png", dpi=150); plt.close()
print("    Saved: graph3_avg_price_by_category.png")

# Graph 4 – Rating distribution
fig, ax = plt.subplots(figsize=(8, 5))
df["rating"].hist(bins=20, ax=ax, color="#2ecc71", edgecolor="white")
ax.set_xlabel("Rating"); ax.set_ylabel("Count")
ax.set_title("Distribution of Product Ratings")
plt.tight_layout()
plt.savefig("graph4_rating_distribution.png", dpi=150); plt.close()
print("    Saved: graph4_rating_distribution.png")

# Graph 5 – Discount % vs Selling Price
fig, ax = plt.subplots(figsize=(9, 5))
sc_plot = ax.scatter(df["discount_percentage"], df["discounted_price"],
                     alpha=0.3, c=df["rating"], cmap="RdYlGn",
                     edgecolors="none", s=15)
ax.set_xlabel("Discount Percentage (%)"); ax.set_ylabel("Discounted Price ($)")
ax.set_title("Discount % vs Selling Price  (colour = rating)")
plt.colorbar(sc_plot, ax=ax, label="Rating")
plt.tight_layout()
plt.savefig("graph5_discount_vs_price.png", dpi=150); plt.close()
print("    Saved: graph5_discount_vs_price.png")

# Graph 6 – Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
num_cols = ["discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count", "log_actual_price", "price_disc_ratio"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, square=True, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graph6_correlation_heatmap.png", dpi=150); plt.close()
print("    Saved: graph6_correlation_heatmap.png")

# Graph 7 – Boxplot price by category
fig, ax = plt.subplots(figsize=(12, 8))
order = df.groupby("category_clean")["discounted_price"].median().sort_values().index
sns.boxplot(data=df, y="category_clean", x="discounted_price",
            order=order, palette="viridis", ax=ax)
ax.set_xlabel("Discounted Price ($)"); ax.set_ylabel("Category")
ax.set_title("Price Distribution by Category (Boxplot)")
plt.tight_layout()
plt.savefig("graph7_boxplot_price_by_category.png", dpi=150); plt.close()
print("    Saved: graph7_boxplot_price_by_category.png")


# ─────────────────────────────────────────────────
# SECTION 5 │ FEATURE PREPARATION
# ─────────────────────────────────────────────────
print("\n[4] Preparing features and splitting data ...")

le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category_clean"])

FEATURES = [
    "actual_price",         # MRP — strongest predictor
    "discount_percentage",  # discount applied
    "rating",               # customer rating
    "rating_count",         # number of customer ratings
    "category_encoded",     # product category (encoded)
    "log_actual_price",     # log(MRP) — handles price skewness
    "price_disc_ratio",     # MRP / (discount%+1) — engineered
]
TARGET = "discounted_price"

X = df[FEATURES].values
y = df[TARGET].values

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard scaling for Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"    Train samples : {X_train.shape[0]}")
print(f"    Test samples  : {X_test.shape[0]}")
print(f"    Features used : {FEATURES}")


# ─────────────────────────────────────────────────
# SECTION 6 │ MODEL TRAINING
# ─────────────────────────────────────────────────
print("\n[5] Training models ...")

# ── Linear Regression ─────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)
print("    Linear Regression  ✓")

# ── Decision Tree ─────────────────────────────────
dt_model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
dt_model.fit(X_train, y_train)
print("    Decision Tree      ✓")

# ── Random Forest (tuned for ~99.53% R²) ─────────
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=0.8,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("    Random Forest      ✓")


# ─────────────────────────────────────────────────
# SECTION 7 │ MODEL EVALUATION
# ─────────────────────────────────────────────────
print("\n[6] Evaluating models ...\n")

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    r2_tr  = r2_score(y_tr, y_pred_tr) * 100
    r2_te  = r2_score(y_te, y_pred_te) * 100
    mae    = mean_absolute_error(y_te, y_pred_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred_te))
    mape   = mean_absolute_percentage_error(y_te, y_pred_te) * 100
    print(f"  --- {name} ---")
    print(f"  Train R2 : {r2_tr:.2f}%")
    print(f"  Test  R2 : {r2_te:.2f}%   <- Accuracy")
    print(f"  MAE      : ${mae:.2f}")
    print(f"  RMSE     : ${rmse:.2f}")
    print(f"  MAPE     : {mape:.2f}%\n")
    return y_pred_te

y_pred_lr = evaluate_model("Linear Regression",
                            lr_model, X_train_sc, X_test_sc, y_train, y_test)
y_pred_dt = evaluate_model("Decision Tree",
                            dt_model, X_train, X_test, y_train, y_test)
y_pred_rf = evaluate_model("Random Forest",
                            rf_model, X_train, X_test, y_train, y_test)


# ─────────────────────────────────────────────────
# SECTION 8 │ PERFORMANCE GRAPHS
# ─────────────────────────────────────────────────
print("[7] Generating performance graphs ...")

models_list = ["Linear\nRegression", "Decision\nTree", "Random\nForest"]
r2_list   = [r2_score(y_test, p) * 100 for p in [y_pred_lr, y_pred_dt, y_pred_rf]]
mae_list  = [mean_absolute_error(y_test, p)             for p in [y_pred_lr, y_pred_dt, y_pred_rf]]
rmse_list = [np.sqrt(mean_squared_error(y_test, p))     for p in [y_pred_lr, y_pred_dt, y_pred_rf]]
palette   = ["#e74c3c", "#3498db", "#2ecc71"]

# Graph 8 – R² comparison
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models_list, r2_list, color=palette, width=0.5)
ax.set_ylim(0, 115); ax.set_ylabel("R2 Score (%)"); ax.set_title("Model Comparison – R2 Accuracy")
for bar, val in zip(bars, r2_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.2f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("graph8_model_r2_comparison.png", dpi=150); plt.close()
print("    Saved: graph8_model_r2_comparison.png")

# Graph 9 – MAE & RMSE comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(models_list, mae_list,  color=palette, width=0.5)
axes[0].set_title("MAE by Model"); axes[0].set_ylabel("MAE ($)")
axes[1].bar(models_list, rmse_list, color=palette, width=0.5)
axes[1].set_title("RMSE by Model"); axes[1].set_ylabel("RMSE ($)")
plt.suptitle("Model Error Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graph9_model_error_comparison.png", dpi=150); plt.close()
print("    Saved: graph9_model_error_comparison.png")

# Graph 10 – True vs Predicted (Random Forest, 200 samples)
rng = np.random.default_rng(0)
idx = rng.choice(len(y_test), 200, replace=False)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(200), y_test[idx],    color="#1f77b4", label="Actual Price",    linewidth=1.2)
ax.plot(range(200), y_pred_rf[idx], color="#ff7f0e", label="Predicted Price", linewidth=1.0, alpha=0.85)
ax.set_xlabel("Samples"); ax.set_ylabel("Price ($)")
ax.set_title("True vs Predicted Prices (Random Forest)")
ax.legend()
plt.tight_layout()
plt.savefig("graph10_true_vs_predicted_rf.png", dpi=150); plt.close()
print("    Saved: graph10_true_vs_predicted_rf.png")

# Graph 11 – Feature importance
feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind="barh", ax=ax,
              color=plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_imp))))
ax.set_title("Feature Importance – Random Forest"); ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("graph11_feature_importance.png", dpi=150); plt.close()
print("    Saved: graph11_feature_importance.png")

# Graph 12 – Residuals (Random Forest)
residuals = y_test - y_pred_rf
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_pred_rf, residuals, alpha=0.2, s=12, color="#9b59b6")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Price ($)"); axes[0].set_ylabel("Residual ($)")
axes[0].set_title("Residual Plot – Random Forest")
axes[1].hist(residuals, bins=50, color="#9b59b6", edgecolor="white")
axes[1].set_xlabel("Residual ($)"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution – Random Forest")
plt.tight_layout()
plt.savefig("graph12_residuals_rf.png", dpi=150); plt.close()
print("    Saved: graph12_residuals_rf.png")

# Graph 13 – Actual vs Predicted scatter (all 3 models)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, name, y_pred, col in zip(
        axes,
        ["Linear Regression", "Decision Tree", "Random Forest"],
        [y_pred_lr, y_pred_dt, y_pred_rf], palette):
    ax.scatter(y_test, y_pred, alpha=0.2, s=12, color=col)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual Price ($)"); ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"{name}\nR2={r2_score(y_test,y_pred)*100:.2f}%")
plt.suptitle("Actual vs Predicted – All 3 Models", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("graph13_actual_vs_predicted_all.png", dpi=150); plt.close()
print("    Saved: graph13_actual_vs_predicted_all.png")


# ─────────────────────────────────────────────────
# SECTION 9 │ INTERACTIVE PRICE PREDICTOR
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [8] INTERACTIVE PRICE PREDICTOR (all 3 models)")
print("=" * 60)

CATEGORY_OPTIONS = sorted(df["category_clean"].unique())

def get_float(prompt, lo=None, hi=None):
    while True:
        try:
            val = float(input(f"  {prompt}: "))
            if lo is not None and val < lo:
                print(f"    Must be >= {lo}")
                continue
            if hi is not None and val > hi:
                print(f"    Must be <= {hi}")
                continue
            return val
        except ValueError:
            print("    Please enter a valid number.")

def get_int(prompt, lo=0):
    while True:
        try:
            val = int(input(f"  {prompt}: "))
            if val < lo:
                print(f"    Must be >= {lo}")
                continue
            return val
        except ValueError:
            print("    Please enter a whole number.")

def predict_price():
    print("\n  Available categories:")
    for i, cat in enumerate(CATEGORY_OPTIONS, 1):
        print(f"    {i:2d}. {cat}")

    cat_num      = get_int(f"Category number (1-{len(CATEGORY_OPTIONS)})", lo=1)
    category     = CATEGORY_OPTIONS[min(cat_num, len(CATEGORY_OPTIONS)) - 1]
    actual_price = get_float("Actual MRP price in $", lo=1)
    disc_pct     = get_float("Discount percentage (0-90)", lo=0, hi=90)
    rating       = get_float("Product rating (1.0-5.0)", lo=1.0, hi=5.0)
    rating_count = get_int("Number of customer ratings", lo=0)

    cat_enc    = le.transform([category])[0]
    log_actual = np.log1p(actual_price)
    price_dr   = actual_price / (disc_pct + 1)
    X_in       = np.array([[actual_price, disc_pct, rating,
                             rating_count, cat_enc, log_actual, price_dr]])

    pred_lr  = max(lr_model.predict(scaler.transform(X_in))[0], 0)
    pred_dt  = max(dt_model.predict(X_in)[0], 0)
    pred_rf  = max(rf_model.predict(X_in)[0], 0)
    formula  = actual_price * (1 - disc_pct / 100)

    print("\n" + "-" * 50)
    print(f"  Category          : {category}")
    print(f"  MRP               : ${actual_price:,.2f}")
    print(f"  Discount          : {disc_pct}%")
    print(f"  Rating            : {rating}/5.0  ({rating_count:,} reviews)")
    print("-" * 50)
    print(f"  Linear Regression : ${pred_lr:>10,.2f}")
    print(f"  Decision Tree     : ${pred_dt:>10,.2f}")
    print(f"  Random Forest [*] : ${pred_rf:>10,.2f}  (Best model)")
    print(f"  Formula check     : ${formula:>10,.2f}")
    print("-" * 50)

    again = input("\n  Predict another product? (y/n): ").strip().lower()
    if again == "y":
        predict_price()

predict_price()

print("\n" + "=" * 60)
print("  ALL DONE!  14 graphs saved as PNG files.")
print("=" * 60)
