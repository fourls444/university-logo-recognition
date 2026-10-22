# Dataset organization

The complete `data/` directory is local-only and excluded from Git, including
collected artwork, PDFs, derived images and source records. Removing it from
the Git index preserves the files on disk. The root manifest remains tracked
as the inventory, but does not contain the image bytes. Existing commits may
still contain these files; ignoring them does not erase Git history.

A fresh clone needs a separately supplied private dataset for training, or
a private model bundle and `artifacts/current.json` for inference. The complete
`artifacts/` directory remains ignored. The project is demonstrated locally
for a portfolio. Dataset integration tests skip when the private data is
absent; synthetic-image tests still run.

On 2026-10-22, 120 legacy logo files moved from `Logo/` into
`data/logos/<university-slug>/`. Neutral names such as
`bangkok-university-legacy-01.png` identify the university and variant without
claiming which version is the original. The 28 sample uploads moved from
`Image/` into `data/examples/`. Each example's university was verified by an
exact decoded RGBA pixel match with a labeled manifest image.

`dataset_manifest.json` and `university_info.csv` remain at the project root.
Candidate collection files are grouped in `data/candidates/`. Application
modules, model versions, feedback, and recovery evidence keep their locations.

The candidate directory contains 29 university folders with neutral names such
as `chulalongkorn-university-logo-candidate-01.png`. Candidate approval status
stays in the source/review records; a readable filename does not imply approval.
The dated augmentation and document collections keep their own `derived/`,
`manifests/`, `previews/`, and `sources/` folders. Collection, extraction, and
augmentation commands are grouped under `scripts/`.

The migration preserves file bytes and SHA-256 hashes, university labels,
source groups, and persisted train/validation assignments. The baseline stays
at 126 training images, 31 validation images, and 30 classes.

## Provenance and reversal

`data/provenance/dataset-manifest-before-organization.json` contains the exact
previous manifest bytes, whose SHA-256 is
`48c7d2b0198fccd65bd252c2804c6e5c3a7f217a7397f861885f7dcc8fc11b81`.
`data/provenance/dataset-path-mapping.json` records every old and new path and
source checksum. The snapshot is historical and its paths describe the former
layout; it is not the active training manifest.

`data/provenance/candidate-records-before-organization/` preserves the original
candidate records and helper sources before updating path references. Payload
images and PDF bytes are unchanged; current JSON/CSV records use the new paths.

Large duplicate raw page caches were moved to the local ignored archive under
`artifacts/archive/github-cleanup-20261022/` when preparing GitHub contents.
They are not runtime inputs; source manifests, review records and image hashes remain.

Existing model metadata keeps its original manifest checksum. Relocating files
changes the active manifest checksum but does not change those models or their
recorded training history. To reverse this migration, move each mapped file
back to its recorded old path, restore the exact snapshot as the active
manifest, and reverse the corresponding current-reader path changes. Verify
checksums and source-group membership before training after any reversal.

Run `python -m pytest tests/test_dataset_organization.py tests/test_dataset.py`
to check byte preservation, manifest metadata, example labels, and dataset
loading. Numeric names inside historical documents or snapshots are retained
as provenance.
