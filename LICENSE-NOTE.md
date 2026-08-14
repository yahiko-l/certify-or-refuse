# Licensing

## Code (`code/`, `data/export_scores_only.py`)
First-party, released under the **Apache License 2.0** (full text in `LICENSE`; attribution
in `NOTICE`). Apache-2.0 was chosen over MIT for its explicit patent grant. The copyright
holder is listed in `NOTICE`.

## Real-data caches (`data/real_cache_*/`: frozen arrays, locks, manifests, access logs)
Derived from third-party corpora — **not yours to relicense**:
- **SQuAD** (source): CC BY-SA 4.0 → attribution + share-alike if you redistribute text.
- **NewsQA** (target): Microsoft Research license → **redistribution restricted**; you most
  likely may ship only **ids + a derive script**, not the article/answer text.

The current release ships text-free frozen eval arrays (scores + losses, no corpus text),
which sidesteps both corpus licenses. See `data/README_DATA.md` for the redistribution and
regeneration options.

## Model weights
Not redistributed. Subject to their own licenses (Meta Llama 3 Community License;
DeepSeek model license). Provenance hashes are in each cache `manifest.json`.
