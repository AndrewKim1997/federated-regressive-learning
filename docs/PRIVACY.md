# Privacy and Data Usage

This repository **must not** include private or contractual datasets (e.g., hospital data like UGEI). Publishing such data would violate privacy laws and contractual agreements.

## Non-Distribution of UGEI

* UGEI (or any similar medical dataset) is **not** included.
* The codebase provides a strict **placeholder interface** at `frl/data/ugei_placeholder.py`.
* To run experiments with private data, users must provide their own *local* dataset path.

## How to Use a Local, User-Owned Dataset

* Recommended format: a single `.npz` file with arrays:

  * `X`: `float32` matrix `[N, D]` or image tensor `[N, H, W, C]`,
  * `y`: integer labels `[N]`.
* Call:

```python
from frl.data.ugei_placeholder import load_ugei_data
X, y = load_ugei_data("/abs/path/to/your_ugei.npz")
```

* No dataset files are uploaded to this repository. `.gitignore` excludes `data/`.

## Prohibited Actions

* Do **not** upload any samples, metadata, or derived artifacts that could re-identify individuals.
* Do **not** attempt to de-identify and redistribute private data via this repository.
* Do **not** circumvent access controls or institutional review processes.

## Recommended Practices

* Keep private data on secured infrastructure under your institution’s policy.
* Review applicable regulations (e.g., HIPAA/GDPR/your local law) before any processing.
* Prefer open datasets (e.g., MNIST/CIFAR-10) for public demonstrations.
