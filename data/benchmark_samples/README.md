# Benchmark Samples — Provenance

Generated: 2026-07-06T12:05:43Z

## Files

| File | Task | Source Dataset | Docs |
|---|---|---|---|
| `translation_de_en.json` | translation | Helsinki-NLP/europarl | 20 |
| `summarisation_en.json` | summarisation | abisee/cnn_dailymail (3.0.0, test split) | 20 |

## Document IDs

### translation

`europarl_de-en_0000`, `europarl_de-en_0001`, `europarl_de-en_0002`, `europarl_de-en_0003`, `europarl_de-en_0004`, `europarl_de-en_0005`, `europarl_de-en_0006`, `europarl_de-en_0007`, `europarl_de-en_0008`, `europarl_de-en_0009`, `europarl_de-en_0010`, `europarl_de-en_0011`, `europarl_de-en_0012`, `europarl_de-en_0013`, `europarl_de-en_0014`, `europarl_de-en_0015`, `europarl_de-en_0016`, `europarl_de-en_0017`, `europarl_de-en_0018`, `europarl_de-en_0019`

### summarisation

`cnn_dm_0000`, `cnn_dm_0001`, `cnn_dm_0002`, `cnn_dm_0003`, `cnn_dm_0004`, `cnn_dm_0005`, `cnn_dm_0006`, `cnn_dm_0007`, `cnn_dm_0008`, `cnn_dm_0009`, `cnn_dm_0010`, `cnn_dm_0011`, `cnn_dm_0012`, `cnn_dm_0013`, `cnn_dm_0014`, `cnn_dm_0015`, `cnn_dm_0016`, `cnn_dm_0017`, `cnn_dm_0018`, `cnn_dm_0019`

## Schema

**translation_de_en.json** — each entry:
```json
{
  "doc_id": "europarl_de-en_0000",
  "source_language": "de",
  "raw_text": "...",
  "source": "europarl",
  "metadata": {"reference_translation": "..."}
}
```

**summarisation_en.json** — each entry:
```json
{
  "doc_id": "cnn_dm_0000",
  "source_language": "en",
  "raw_text": "...",
  "source": "cnn_dailymail",
  "metadata": {"reference_summary": "...", "cnn_id": "..."}
}
```

These files are committed to git. Re-run this script after `python main.py` downloads larger samples.
