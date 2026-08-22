# Security Test Report

## Test date

2026-08-23

## Scope

The Python library, CLI entry point, package metadata, and repository source.
No physical GPU or remote service was targeted.

## Checks performed

- Invalid type and oversized allocation inputs were rejected.
- Cross-allocator, forged, and double-free handles were rejected.
- Concurrent allocate and free operations preserved accounting invariants.
- CLI JSON output was parsed without executing content.
- Trace retention was bounded and returned through defensive copies.
- Benchmark JSON output was parsed and its total resident memory comparison was verified.
- Source was reviewed for `eval`, `exec`, dynamic imports, shell calls,
  sockets, HTTP clients, credential reads, and unsafe deserialization.
- The 17-test suite passed locally.
- GitHub Actions ran the test suite on Python 3.10, 3.11, 3.12, and 3.13.

## Result

Pass for the defined local threat model. The product remains a simulator and
must not be presented as a real CUDA allocator or used as a security boundary.
