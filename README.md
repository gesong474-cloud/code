# GFP fluorescence fitness

Stage 1 describes the measured GFP substitution landscape and prepares reproducible tables for subsequent experiments. It covers quality control and exploratory data analysis of the complete assay.

## Data

- [ProteinGym v1.3 downloads](https://proteingym.org/download). The DMS Substitutions archive supplies `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`. The DMS reference table supplies the matching GFP row in `data/raw/DMS_substitutions.csv`. The download release is v1.3 and the version field on that reference row is 0.1.
- Sarkisyan et al. (2016), *Local fitness landscape of the green fluorescent protein*, [Nature, doi:10.1038/nature17995](https://doi.org/10.1038/nature17995). ProteinGym records a FACS fluorescence assay of *Aequorea victoria* GFP, with phenotype `mean_medianBrightness_per_aaseq` and directionality +1.

## Reproduce

The commands run from the repository root. The verified environment uses Python 3.12.7. Python 3.11 and 3.12 are compatible with the pinned dependencies.

```sh
python -m venv ../gfp-stage1-env
```

Activate the environment in PowerShell with `..\gfp-stage1-env\Scripts\Activate.ps1` or in bash with `source ../gfp-stage1-env/bin/activate`. Then run:

```sh
python -m pip install -r requirements.txt
python src/data_processing.py
python src/eda.py
```

Both scripts accept an optional `--root PATH` for a directory with the same layout. They read the raw inputs in place and regenerate the processed tables and EDA outputs.

## Methods

- Each variant ID is assigned in source row order. The quality report maps flagged IDs to source CSV line numbers and gives the reason and observed values. A failed quality check stops processing after the report is written. All passing records are retained in source order.
- Substitution positions are 1-based. The pipeline validates each substitution against the 238-residue reference and rebuilds each sequence for comparison. The processed CSV keeps each input DMS score verbatim. In the variant table, semicolon-separated positions and residue columns follow the order in `mutant`.
- A binary label of 1 uses `DMS_score >= 2.5`, the ProteinGym cutoff for this assay. The source CSV has no measured wild-type row. The FASTA records the reference sequence, and the quality report records the absence of a measured wild-type fitness.
- The long table stores one substitution occurrence per row and links it to the variant ID. Fitness stays at the variant level. Position mutation frequency is the fraction of all variants that carry a substitution at that position.
- The heatmap uses only records with one substitution. It retains `NA` for missing cells, including reference amino acid identities. Descriptive statistics use sample standard deviation and linear quantiles. The row-order analysis uses Pearson correlation, tie-aware Spearman correlation, and contiguous quartile blocks.

## Main results

The 51,714 variants yield 200,547 substitution occurrences. There are 1,084 independently measured single substitutions, 1,810 distinct substitutions across all variants, and 233 of 238 positions covered. Variants have 1 through 15 substitutions, with 35,584 having 1 through 4. The median DMS score is 3.2873. A total of 30,086 variants (58.18%) meet the cutoff. The original rows have an upward fitness trend (Spearman rho = 0.290), and their scores are nonmonotonic.

## Outputs

| File | Contents |
|---|---|
| `data/processed/gfp_wildtype.fasta` | Reference GFP sequence |
| `data/processed/gfp_metadata.json` | Assay metadata, literature, input versions and hashes |
| `data/processed/gfp_processed.csv` | Standardized, lossless variant table |
| `data/processed/gfp_mutations_long.csv` | Individual substitution occurrences, linked by variant ID |
| `results/eda/data_quality_report.json` | Row-level checks, assay counts, and wild-type measurement status |
| `results/eda/eda_summary.json` | Fitness and label statistics, candidate counts, coverage, single-substitution statistics, and source row-order assessment |
| `results/eda/fitness_by_mutation_count.csv` | Variant counts, proportions, fitness summaries, and threshold rates by mutation count |
| `results/eda/position_mutation_frequency.csv` | Mutation frequencies and substitution counts at every reference position |
| `results/eda/single_substitution_heatmap.csv` | Measured single-substitution fitness matrix |
| `results/eda/figure_1_dms_score_distribution.png` | Equal-width fitness histogram |
| `results/eda/figure_2_mutation_count_distribution.png` | Mutation count histogram |
| `results/eda/figure_3_fitness_by_mutation_count.png` | Fitness boxplots grouped by mutation count |
| `results/eda/figure_4_fitness_threshold_by_mutation_count.png` | Fraction at or above the ProteinGym cutoff by mutation count |
| `results/eda/figure_5_position_mutation_frequency.png` | Frequency of mutation at each reference position |
| `results/eda/figure_6_single_substitution_heatmap.png` | Measured single-substitution fitness with unmeasured cells in gray |

All six figures are PNGs at 300 DPI.
