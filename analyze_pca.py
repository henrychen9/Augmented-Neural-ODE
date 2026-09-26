import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------------------
# load sweep data
# ----------------------------

DATA_PATH = "sweep_degree2.pt"

obj = torch.load(DATA_PATH, map_location="cpu")
records = obj["records"]

# ----------------------------
# extract data
# ----------------------------

X = []
labels = []

for r in records:
    init_w = r["init_weights"]

    flat_parts = []

    # sort keys to keep layer order: linear_0, linear_1, ...
    for k in sorted(init_w.keys()):
        if k.endswith(".W"):        # weights only (skip biases for now)
            flat_parts.append(init_w[k].reshape(-1).numpy())

    full_vec = np.concatenate(flat_parts)
    X.append(full_vec)

    # failed vs non-failed only
    labels.append(r["label"] == "failed")

X = np.stack(X, axis=0)   # [N, 240]
labels = np.array(labels)

print("data matrix shape:", X.shape)
print("failed runs:", labels.sum())
print("non-failed runs:", (~labels).sum())

# ----------------------------
# PCA to 3D
# ----------------------------

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print("explained variance ratio:", pca.explained_variance_ratio_)
print("total explained:", pca.explained_variance_ratio_.sum())

# ----------------------------
# visualize
# ----------------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# non-failed
ax.scatter(
    X_pca[~labels, 0],
    X_pca[~labels, 1],
    X_pca[~labels, 2],
    c="blue",
    alpha=0.3,
    s=20,
    label="non-failed"
)

# failed
ax.scatter(
    X_pca[labels, 0],
    X_pca[labels, 1],
    X_pca[labels, 2],
    c="red",
    alpha=0.9,
    s=40,
    label="failed"
)

ax.set_title("PCA of Initial W1 (degree=2)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# cumulative explained variance
# ----------------------------

# fit PCA with all possible components
pca_full = PCA()
pca_full.fit(X)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# find minimum dimensions for common thresholds
thresholds = [0.90, 0.95, 0.99]
for t in thresholds:
    k = np.argmax(cumulative >= t) + 1
    print(f"{int(t*100)}% variance explained at {k} components")

# plot
plt.figure(figsize=(7,5))
plt.plot(
    np.arange(1, len(cumulative)+1),
    cumulative,
    marker='o'
)

plt.axhline(0.90, linestyle='--')
plt.axhline(0.95, linestyle='--')
plt.axhline(0.99, linestyle='--')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained vs Number of PCs")
plt.grid(True)
plt.tight_layout()
plt.show()
