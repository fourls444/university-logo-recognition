# Training image variants

Run `python dataset_augmentation.py` from the repository root.

- `derived/<university>/`: grayscale, ±6° rotation, dim light, JPEG compression and scan-like blur.
- `manifests/manifest.json`: parent image, label, source family, hashes and transformation.
- `previews/preview.jpg`: representative visual check.

One parent per training family prevents repeatedly multiplying existing near-duplicates. Known held-out families and exact canonical pixel matches are excluded. Derivatives retain the parent's training family; they are not independent new source images.

These files remain candidates. Generation does not modify the active dataset, train a model or establish an accuracy improvement. Training already applies some augmentation; compare approaches before importing these too, to avoid overweighting one source family.
