# 🧠 KMeans Clustering Project

This project implements and compares multiple clustering algorithms using a clean, modular Python structure. It supports:

- Standard k-Means
- DBSCAN
- Spherical k-Means (patched for scikit-learn compatibility)

---

## 📌 Deliverable 1: Well-Documented and Modular Source Code

### 📂 Structure

- `main.py`: Orchestrates loading datasets, preprocessing, and running all clustering algorithms.
- `kmeans.py`: Custom implementation of the k-Means algorithm from scratch.
- `clustering.py`: Contains logic for DBSCAN and patched Spherical KMeans.
- `utils.py`: Normalization, dimensionality reduction, plotting, and utility functions.
- `plots/`: Automatically generated visualizations for each dataset/algorithm combination.

Each module is self-contained and documented to support modular understanding and future extensibility.

---

## ⚙️ Environment Setup

To ensure compatibility, use **Python 3.9** and avoid Anaconda due to licensing restrictions.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AbdJuma/Kmeans.git
   cd Kmeans
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python3.9 -m venv kmeans_env
   source kmeans_env/Scripts/activate  # Git Bash or WSL
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧩 Patching `spherecluster`

To use **Spherical KMeans**, apply the following patch:

- 📦 Download: [spherecluster_patched.zip](https://github.com/AbdJuma/Kmeans/blob/master/kmeans_env/Lib/site-packages/spherecluster/spherecluster_patched.zip)
- 📁 Replace the content of:
  ```bash
  kmeans_env/Lib/site-packages/spherecluster/
  ```
  with the unzipped patched files.

---

## ▶️ Running the Code

From the root directory:

```bash
python main.py
```

This script:
- Loads datasets (Iris, Moons, Circles, etc.)
- Normalizes and reduces them to 2D
- Runs k-Means, DBSCAN, and Spherical k-Means
- Saves the visualizations into `plots/`

---

## 💻 Running in Visual Studio 2022

- Open the solution folder in Visual Studio.
- Right-click `main.py` → **Set as Startup File**.
- Run the script with the activated `kmeans_env`.

---

## 🧪 Notes

- Matplotlib uses the `'agg'` backend for compatibility; visual outputs are saved, not shown.
- All clustering outputs are visualized for direct inspection.

---
