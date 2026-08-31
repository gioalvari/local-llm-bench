# Dataset contract

The repository includes a small, source-grounded EIA smoke benchmark to verify
the evaluation pipeline. It is not a sealed test set and must not be used to
claim fine-tuning improvements. A production training study must version its
source documents and generated QA records independently while retaining
provenance and licensing information.

One JSON object per line is expected:

```json
{
  "item_id": "market-rules-001",
  "source_document_id": "document-2025-01",
  "source_url": "https://example.org/source",
  "source_revision": "2025-01",
  "category": "settlement",
  "context": "Source-grounded excerpt",
  "question": "What is the settlement interval?",
  "answer_type": "numeric",
  "canonical_answer": "15 minutes",
  "accepted_aliases": ["15 min"],
  "expected_value": 15,
  "accepted_units": ["minutes", "min"],
  "absolute_tolerance": 0.0,
  "evidence_spans": ["settlement interval is 15 minutes"],
  "split": "smoke"
}
```

Split by source document, publication period, or topic group. Run duplicate and
near-duplicate detection before sealing the test split. Test answers must not be
used for prompt design, threshold selection, training, or checkpoint selection.

## Included smoke benchmark

`energy/eia-electricity-smoke.jsonl` contains 12 short questions across units,
generation, grid operation, and energy storage. Every item includes the exact
supporting excerpt and public EIA URL. The source revision records the access
date because these explanatory web pages are updated in place.

Sources:

- https://www.eia.gov/energyexplained/electricity/measuring-electricity.php
- https://www.eia.gov/energyexplained/electricity/electricity-in-the-us.php
- https://www.eia.gov/energyexplained/electricity/delivery-to-consumers.php
- https://www.eia.gov/energyexplained/electricity/energy-storage-for-electricity-generation.php
