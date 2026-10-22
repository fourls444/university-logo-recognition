# Additional dataset audit — 2026-10-22

**0 additional training images approved or downloaded.** The active data and model were not changed.

- `manifests/dataset-audit.json`: source-level license/class audit and search scope.
- `manifests/metadata-candidates.json`: 15 metadata candidates, all explicitly ineligible for training. 13 names exactly match current labels; the corrupted KMITL apostrophe is mapped explicitly; Prince of Songkla is outside the current labels.
- `sources/world-universities-thailand-metadata.json`: the bounded Thailand subset downloaded from the MIT metadata repository. Includes an MFU campus image and missing image URLs; it is not a labeled logo image dataset.
- `sources/world-universities-LICENSE` and `sources/world-universities-README.md`: upstream provenance. Metadata license does not establish rights in external university artwork.
- `sources/download-attempts.json`: the CC BY-SA 4.0 SWU emblem was unavailable (HTTP 403); no attempts to bypass the response.
- `images/` and `previews/` are empty because no image passed acquisition, license, label, visual-review and split-family checks.

The 186-image CC BY 4.0 President University dataset is real, but all nine labels are Indonesian universities outside this model. OpenLogo is a possible separate pretraining topic, with different brand labels and incomplete image-license evidence; neither belongs in this classifier's training manifest.

No SHA or perceptual score is invented for unavailable images. Before accepting any future candidate, decode and visually review it, compute SHA-256 and perceptual similarity to all existing images, and conservatively group any matching held-out artwork with its held-out source family. A new website or file format alone does not create an independent validation/training family.
