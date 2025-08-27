# docs/SCENARIOS.md

# Scenario Definitions

We provide three canonical, programmatically generated scenarios for federated experiments. Each scenario produces:

* `indices_*.json`: client index lists,
* `summary_*.csv`: per-client sample counts and class distributions,
* `config_echo_*.yaml`: echoed configuration with timestamp.

Run:

```bash
python -m frl.scenarios.scenario_gen -c <path-to-scenario.yaml> -o results/scenarios --preview
```

---

## S1 — Equal Distribution, Different Sizes

**Intent:** All clients share the *same class distribution* as the global dataset; only sample sizes differ.

**Definition**

* Let global class distribution be $p$.
* For client $i$, draw $n_i$ samples using stratified sampling according to $p$.
* Sizes $\{n_i\}$ are specified in YAML.

**Reference YAML:** `frl/scenarios/s1_equal_dist_diff_size.yaml`

---

## S2 — Heterogeneous Distributions, Different Sizes

**Intent:** Clients have *different* class distributions and (optionally) different sizes.

Two modes:

1. **Dirichlet non-IID**

   * For each class, draw client proportions from $\text{Dirichlet}(\alpha)$.
   * Smaller $\alpha$ ⇒ more skew.
   * YAML key: `scenario.alpha`.

2. **Fixed per-client proportions**

   * Provide `scenario.class_props: [M x K]` where each row sums to 1.
   * The generator allocates integer counts per class to match these targets.

**Reference YAML:** `frl/scenarios/s2_hetero_dist_diff_size.yaml`

---

## S3 — Class-Missing Scenario

**Intent:** Some clients *lack* one or more classes.

**Procedure**

* Start from `iid` or `dirichlet` base split.
* For each client $i$, remove all samples of classes in `missing_map[i]`.
* Optionally set `fill_to_target: true` to top-up with leftover samples to keep sizes unchanged.

**Reference YAML:** `frl/scenarios/s3_class_missing.yaml`

---

## Example Snippets

**S2 with fixed proportions**

```yaml
scenario:
  type: s2_hetero_dist_diff_size
  class_props:
    - [0.40, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    - [0.05, 0.40, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    - [0.05, 0.05, 0.40, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    - [0.05, 0.05, 0.05, 0.40, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05]
    - [0.20, 0.05, 0.05, 0.05, 0.40, 0.05, 0.05, 0.05, 0.05, 0.05]
```

**S3 with missing classes**

```yaml
scenario:
  type: s3_class_missing
  base: iid
  missing_map:
    0: [0, 1]
    1: [2, 3]
    2: [4, 5]
    3: [6]
    4: [7, 8, 9]
  fill_to_target: false
```
