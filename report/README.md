# WorldLedger technical report

September 23, 2026. Both language editions contain six figures.

- [English manuscript](technical-report.md) and [PDF](WorldLedger-technical-report.pdf)
- [中文稿](technical-report-zh.md)与 [PDF](WorldLedger-technical-report-zh.pdf)
- [Claim-source index](claim-source-index.json) and [evidence index](report-evidence-index.json)

The figures cover the system and distribution boundary, verification decisions, control/state timing and dataset views, conceptual obstacle-transfer candidates, separate case counts, and contact-derived musical events. Each has an English and Chinese SVG in `figures/`. Concept diagrams are labeled as such; case counts are read from the public JSON summaries in `examples/`.

The report distinguishes included source, external services, case evidence, and architectural extensions. The full transaction controller, teacher production archive, and audiovisual executable are external dependencies.

## Build

From the repository root, install ReportLab and provide a CJK TrueType font that ReportLab can embed:

```bash
python3 -m pip install reportlab
export WORLDLEDGER_CJK_FONT="/path/to/cjk-font.ttf"
python3 report/build_report.py
```

On macOS, the builder also checks the conventional system location for Arial Unicode. No font binary is distributed here. PDF fonts are embedded; SVGs use platform font fallbacks.

The builder regenerates both PDFs, all twelve SVGs, and `build-manifest.json`. The manifest records source and PDF hashes; it is not an experiment validation receipt. Rendering and translation checks in the other JSON files refer to the reviewed release and must be refreshed after editing.

For visual review, render both PDFs with Poppler:

```bash
mkdir -p tmp/pdfs
pdftoppm -scale-to 1200 -png report/WorldLedger-technical-report.pdf tmp/pdfs/en
pdftoppm -scale-to 1200 -png report/WorldLedger-technical-report-zh.pdf tmp/pdfs/zh
```

No DOI has been registered. Author metadata and the final public license remain pending.
