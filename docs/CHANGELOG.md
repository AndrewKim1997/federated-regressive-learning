# Changelog

All notable changes to this project will be documented in this file.
This project adheres to *Keep a Changelog* and uses semantic versioning.

## [0.1.0] - 2025-08-27

### Added

* Initial public **method-reproduction** release of *federated-regressive-learning*.
* Scenario generator (`frl/scenarios/scenario_gen.py`) with S1/S2/S3 YAMLs.
* NumPy-based federated runner and plotting scripts.
* Dataset loaders (MNIST via scikit-learn; CIFAR-10 via torchvision), transform utilities.
* Documentation: `REPRODUCIBILITY.md`, `SCENARIOS.md`, `PRIVACY.md`.
* CI workflow, Dockerfile, environment specs.

### Notes

* Exact replication of the 2024 paper’s conditions is not guaranteed; scenarios are seeded and programmatic to reproduce **relative trends**.
* UGEI or any private dataset is never distributed; only a placeholder interface is included.

[0.1.0]: https://github.com/AndrewKim1997/federated-regressive-learning/releases/tag/v0.1.0 "to be created"
