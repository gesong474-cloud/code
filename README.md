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

## Stage 2 — dataset splits

Stage 2 uses the full, quality-checked set of 51,714 variants. The four splits support distinct experiment roles:

| Split file | Variants | Role |
|---|---:|---|
| `train_initial.csv` | 3,000 | Shared initial experimental observations |
| `validation.csv` | 1,000 | Model selection and hyperparameter tuning |
| `test.csv` | 5,000 | Independent evaluation of predictor generalization within this assay |
| `candidate_pool.csv` | 42,714 | Unlabeled sequence proposals for virtual experiments |

Scikit-learn draws the initial training set, validation set and independent test set in that order from the remaining variants. The remainder forms the candidate pool. Each draw uses only mutation-count group (1, 2, 3, 4, 5, 6–7, 8–15). This preserves the representation of mutation loads while withholding fitness and binary labels from the partitioning decision. Seed 42 was specified in advance. Independent random streams shuffle the rows of each split after assignment.

### Identifiers and access

- The experiment ID is the hexadecimal SHA-256 hash of the complete ASCII sequence. Assignment begins in lexicographic hash order and is independent of the input row order. Use the ID for joins only. Keep it out of model features.
- The candidate table uses an explicit allowlist of sequence and mutation fields, along with the experiment ID. `data/splits/oracle_labels.csv` holds candidate fitness scores and has an independently randomized row order. Join only by experiment ID inside the future Virtual Experiment / Oracle module.
- `data/splits/split_manifest.csv` records the original-to-experiment ID mapping and split membership for reproducibility. Keep it out of model and Agent inputs. Keep the raw and processed source data, all EDA and split audits, Oracle labels, validation, and test data outside the Agent's accessible workspace. The repository organizes files by role. The future runner must enforce access through controlled Python functions and provide only permitted observations and experiment feedback.
- The independent test set is for final model evaluation. Keep its sequences and labels out of training, candidate scoring, virtual experiments and Agent decisions. This in-assay random split permits related mutational backgrounds across splits.

### Reproduce Stage 2

Use the Python environment described above. From the repository root:

```sh
python -m pip install -r requirements.txt
python src/data_split.py --seed 42 --train-size 3000 --validation-size 1000 --test-size 5000 --candidate-size 42714
python -m unittest discover -s tests -v
```

The four sizes must sum to 51,714. Omit `--candidate-size` to use the remaining count. Custom draw sizes must satisfy scikit-learn's per-group stratification constraints. The script checks the Stage 1 quality report and validates staged CSVs before publishing them. It verifies exact counts, disjointness, preservation of source values, candidate-to-Oracle correspondence, and reproducibility.

SHA-256 records the exact bytes read and written by a run. Checkout line-ending conversions can change input hashes, and rendering environments can change figure hashes.

### Stage 2 results

| Split | Samples | Mean fitness | Median fitness | Sample SD | Fitness ≥ 2.5 |
|---|---:|---:|---:|---:|---:|
| Initial training | 3,000 | 2.6516 | 3.3019 | 1.0635 | 57.77% |
| Validation | 1,000 | 2.6251 | 3.2371 | 1.0565 | 56.80% |
| Independent test | 5,000 | 2.6573 | 3.2783 | 1.0507 | 58.10% |
| Candidate pool | 42,714 | 2.6592 | 3.2900 | 1.0592 | 58.25% |

All four subsets contain all seven stratification groups. The largest difference in group share versus the full assay is 0.052 percentage points. Validation has the largest observed shift in threshold rate, 1.38 points below the full assay. All five 14-substitution records and both 15-substitution records are retained. The rare exact mutation counts need not occur in every split.

`results/splits/split_summary.json` contains the full offline audit, including fitness quantiles, integrity checks, random states, and SHA-256 hashes. `split_distribution.csv` contains exact counts and group proportions. The same directory holds two 300 DPI figures for the normalized mutation-count and fitness distributions. Keep these audits, including the candidate-pool fitness summaries, outside the Agent's inputs.

### Parameters reserved for virtual evolution

The future experiment starts with 3,000 measured samples, uses 3 evolution rounds of 20 experiments each (60 additional experiments total), and uses primary seed 42. The Oracle returns experimentally measured DMS scores. Random Mutation, Model-only, LLM Agent, and Knowledge-enhanced LLM Agent share the same initial, validation, test and candidate splits and budget. Each method maintains its own experiment history.
