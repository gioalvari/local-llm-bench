# Dataset contract

Training and evaluation data are not bundled with this repository. Public
source documents and generated QA records must be versioned independently and
must retain provenance and licensing information.

One JSON object per line is expected:

```json
{
  "item_id": "market-rules-001",
  "source_document_id": "document-2025-01",
  "source_revision": "2025-01",
  "category": "settlement",
  "context": "Source-grounded excerpt",
  "question": "What is the settlement interval?",
  "answer_type": "numeric",
  "canonical_answer": "15 minutes",
  "accepted_aliases": ["15 min"],
  "expected_unit": "minutes",
  "absolute_tolerance": 0.0,
  "evidence_spans": ["settlement interval is 15 minutes"],
  "split": "test"
}
```

Split by source document, publication period, or topic group. Run duplicate and
near-duplicate detection before sealing the test split. Test answers must not be
used for prompt design, threshold selection, training, or checkpoint selection.
