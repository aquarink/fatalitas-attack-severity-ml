# Fatalitas

Eksperimen klasifikasi tingkat fatalitas serangan berbasis teks dengan fokus pada reproduksi baseline notebook, perbandingan model klasik, ensemble, dan pendekatan alternatif yang tetap dekat dengan resep eksperimen awal.

## Isi Proyek

- `Fatalitas_Hybrid_ML_NLP_XGBoost_SVM_V2.ipynb`
  Notebook asli eksperimen.
- `reproduce_ipynb_baseline.py`
  Reproduksi baseline notebook sedekat mungkin dari terminal.
- `search_ipynb_baseline_holdout.py`
  Pencarian ringan di sekitar resep notebook untuk menyamai atau menurunkan error rate baseline.
- `search_alternative_baselines.py`
  Perbandingan model dan kombinasi lain dengan preprocessing dan split yang sama seperti baseline notebook.
- `run_best_alternative_baseline.py`
  Menjalankan langsung baseline alternatif terbaik saat ini, yaitu `LinearSVC`.
- `sweep_focused_alternatives.py`
  Sweep terfokus di sekitar `LinearSVC`, `ComplementNB`, dan ensemble ringan untuk mengejar error rate yang lebih rendah.
- `fatalitas_core.py`
  Utilitas bersama untuk preprocessing, evaluasi, dan helper model.
- `run_single_models.py`
  Kajian komparatif model tunggal.
- `run_ensemble_models.py`
  Kajian komparatif model ensemble.
- `run_bert_baseline.py`
  Baseline BERT opsional.
- `run_accuracy_optimization.py`
  Eksperimen pencarian konfigurasi yang lebih berorientasi accuracy.
- `results/`
  Folder hasil eksperimen yang dihasilkan script.

## Dataset

File utama yang digunakan:

- `MtblDataSerAsSelPakistan2425.xlsx`

Kolom target dibentuk dari `KorbanTewas` menjadi 4 kelas:

- `0-2 Rendah`
- `3-5 Sedang`
- `6-7 Tinggi`
- `8+ Sangat Tinggi`

## Tujuan Eksperimen

Proyek ini memiliki dua fokus utama:

1. Mereproduksi hasil notebook awal sebagai baseline.
2. Membandingkan model lain untuk melihat apakah error rate dapat dipertahankan atau diturunkan dengan setup yang valid.

## Model yang Dicoba

### Baseline notebook

- `SVM RBF + XGBoost (soft voting)`

### Model alternatif

- `Complement Naive Bayes`
- `LinearSVC`
- `Calibrated LinearSVC`
- `Logistic Regression`
- `XGBoost`
- `RandomOverSampler + ComplementNB`
- `Soft voting` beberapa kombinasi model
- `Ordinal / regression-based approach`
- `BERT` opsional sebagai komparasi tambahan

## Hasil Awal

Beberapa hasil holdout awal yang sudah muncul pada setup mirip notebook:

- `LinearSVC`: error rate `14.29%`
- `ComplementNB`: error rate `15.71%`
- `SVM + XGBoost ensemble`: error rate `15.71%`

Catatan:

- Angka ini berasal dari satu holdout split yang sama dengan notebook.
- Untuk evaluasi yang lebih ketat tetap disarankan membandingkan juga dengan cross-validation.

## Kebutuhan

Disarankan menggunakan Python 3.9+.

Install dependency utama:

```bash
python3 -m pip install pandas numpy scikit-learn imbalanced-learn openpyxl nltk xgboost
```

Jika ingin menjalankan baseline BERT:

```bash
python3 -m pip install transformers datasets accelerate torch
```

Pada macOS, XGBoost mungkin memerlukan OpenMP:

```bash
brew install libomp
```

## Cara Menjalankan

Masuk ke folder proyek:

```bash
cd "/Volumes/SSD 850/PROJECTS/fatalitas"
```

### 1. Reproduksi baseline notebook

```bash
python3 reproduce_ipynb_baseline.py
```

Output metrik akan disimpan ke:

- `results/ipynb_baseline_metrics.json`

### 2. Search ringan di sekitar baseline notebook

```bash
python3 search_ipynb_baseline_holdout.py
```

Output:

- `results/ipynb_baseline_search_results.csv`
- `results/ipynb_baseline_best_config.json`

### 3. Coba baseline alternatif dengan resep notebook yang sama

```bash
python3 search_alternative_baselines.py
```

Output:

- `results/alternative_baselines_results.csv`
- `results/alternative_baselines_best.json`

### 4. Jalankan baseline alternatif terbaik saat ini

```bash
python3 run_best_alternative_baseline.py
```

Output:

- `results/best_alternative_baseline_metrics.json`
- `results/best_alternative_baseline_report.txt`

### 5. Sweep terfokus untuk mencoba menurunkan error rate lebih jauh

```bash
python3 sweep_focused_alternatives.py
```

Output:

- `results/focused_alternative_sweep_results.csv`
- `results/focused_alternative_sweep_best.json`

### 6. Jalankan kajian komparatif model tunggal

```bash
python3 run_single_models.py
```

Output:

- `results/single_models_results.csv`

### 7. Jalankan kajian komparatif ensemble

```bash
python3 run_ensemble_models.py
```

Output:

- `results/ensemble_models_results.csv`

### 8. Jalankan baseline BERT opsional

```bash
python3 run_bert_baseline.py
```

Output:

- `results/bert_baseline_results.csv`

## Catatan Metodologis

- Dataset relatif kecil dan imbalanced.
- Accuracy tinggi tidak selalu berarti model baik di semua kelas.
- Baseline notebook direproduksi sebagai referensi holdout, lalu dibandingkan dengan baseline alternatif yang memakai resep preprocessing dan split yang sama.
- Sweep terfokus pada `LinearSVC`, `ComplementNB`, dan ensemble ringan disediakan untuk mencari konfigurasi holdout yang lebih rendah dari baseline awal.
- Untuk pelaporan akademik, gunakan juga:
  - `macro F1`
  - `balanced accuracy`
  - `quadratic kappa`
- Beberapa eksperimen yang mengejar error rate rendah pada holdout belum tentu menjadi model terbaik secara keseluruhan.

## Saran Nama Repo

Beberapa opsi nama repo yang cukup rapi:

- `fatalitas-text-classification`
- `fatalitas-attack-severity-ml`
- `fatalitas-fatality-prediction`
- `fatalitas-svm-xgboost-comparison`

## Saran Deskripsi Repo

Opsi deskripsi singkat:

- `Text-based fatality level classification with SVM, XGBoost, Naive Bayes, ensemble, and comparative baselines.`
- `Comparative machine learning experiments for attack fatality prediction from incident narratives.`
- `Reproducible baseline and comparative study for fatality-level text classification.`
