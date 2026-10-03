# Data

## Sample corpus (included)

`data/sample/contracts.json` is a small synthetic corpus: 14 French public
contracts with realistic fields (acheteur, objet, montant, titulaire, CPV,
procédure…). It's deliberately small so the eval suite runs in seconds and
works offline. Every question in `evals/questions.jsonl` is grounded in it.

## Using real DECP data

The real data lives on [data.economie.gouv.fr](https://www.data.economie.gouv.fr),
dataset "DECP" (données essentielles de la commande publique). One JSON per
contract, same fields as the sample plus a few extras.

To adapt the pipeline:

1. Download the yearly DECP JSON dump and drop it in `data/decp/`
   (gitignored — it's hundreds of MB).
2. Write a loader that maps DECP records to `src.ingest.Document`
   (see `load_sample_corpus` for the 10-line pattern).
3. Filter to a subset first (one buyer, one year). The full dump will want
   an IVF index instead of IndexFlatIP — see the note in `src/store.py`.

I kept the sample corpus separate on purpose: evals must stay fast and
reproducible, and real data changes under your feet.
