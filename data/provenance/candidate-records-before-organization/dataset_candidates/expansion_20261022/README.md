# Official document logo candidates

- `sources/documents/`: original [Kasetsart University identity manual](https://www.ku.ac.th/web-upload/m_document/KSU_Manual.pdf), downloaded from the university website.
- `derived/kasetsart-university/`: nine reviewed crops: Thai/English seals, primary/black/reversed logos, and four background colors.
- `manifests/sources.json`: source URL, document hash, page, crop coordinates and family.
- `previews/`: rendered pages and the crop contact sheet.

Run `python dataset_candidates/expansion_20261022/extract_ku_identity.py` to reproduce the crops from the saved PDF. Pages are rendered locally with pypdfium2.

The official KU branding family already belongs to validation. These related crops therefore remain validation candidates, **not training additions**. They provide branding coverage but are not independent camera or scan samples. The active manifest and model are unchanged.

The [official Mahidol identity PDF](https://mahidol.ac.th/wp-content/uploads/document/logo_mahidol/MU-Visual-Identity-Guideline01.pdf) returned HTTP 403 and was not collected. Public access does not establish redistribution permission for university artwork.
