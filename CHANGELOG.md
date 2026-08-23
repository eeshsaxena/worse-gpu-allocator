# Changelog

## 0.2.0

- Added strict input validation and a configurable request-size limit.
- Added ownership checks for forged, cross-allocator, and double-free handles.
- Added thread-safe state management and JSON CLI output.
- Added a formal SRS, security policy, and security test report.

## 0.3.0

- Added bounded allocation tracing with defensive copies and clearing.
- Added a reference allocator and comparison benchmark.
- Added the `worse-gpu-benchmark` command and benchmark JSON output.
- Added regression tests for tracing and benchmark reporting.
- Added total resident memory to benchmark comparisons so CPU fallback is visible.

## 0.4.0

- Added public invariant validation for integration tests and teaching tools.
- Added randomized workload tests covering 4,000 allocator operations.
- Added atomic strict OOM regression coverage.
- Added safe CLI rejection for zero and negative workload sizes.
- Fixed a reference allocator cross-free state corruption edge case.
- Hardened CI with current Node 24 action lines, read-only permissions, and a timeout.

## 0.5.0

- Added deeper accounting invariants and a hard trace retention limit.
- Rejected non-finite CLI delays such as `NaN` and `Infinity`.
- Expanded adversarial and randomized testing coverage.
- Fixed stale module version metadata before distribution.

## 0.6.0

- Added a 100,000 step cap to CLI and programmatic benchmark workloads.
- Added regression coverage for runaway workload rejection.
- Fixed module and package version metadata to `0.6.0`.

## 0.1.0

- Added a seeded GPU and CPU allocation simulator.
- Added intentional fragmentation, waste, and forgotten frees.
- Added strict GPU exhaustion mode.
- Added installable CLI entry point.
- Added tests, packaging metadata, MIT license, and CI-ready project layout.
