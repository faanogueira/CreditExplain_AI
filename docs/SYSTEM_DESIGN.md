# System Design

## Components

### Risk engine

A conventional model, scorecard or externally supplied score calculates the risk signal. The demo implementation provides a deterministic proxy only for integration testing.

### Context builder

Maps raw model features to approved reason codes. Raw data is minimized before it reaches the LLM.

### Explanation LLM

Fine tuned with SFT and DPO. It receives structured context and generates a JSON compatible explanation.

### Validator

Checks schema, reason code coverage, unsupported numeric claims and prohibited terms. A failed validation blocks the response from being marked as production ready.

### Preference judge

Fine tuned on pairwise human preference data. It predicts A, B or tie and supports regression tests for style quality.

### Observability

Captures latency, model version, adapter version, prompt schema version, validation status and aggregate quality metrics. Application data should be logged according to the institution's retention policy.

## Offline promotion flow

```text
new adapter
   |
   v
static validation suite
   |
   v
preference benchmark
   |
   v
position and length bias audit
   |
   v
human review sample
   |
   v
model registry
```
