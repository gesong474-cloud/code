"""Validate and standardize the ProteinGym GFP substitution assay."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from textwrap import fill

ASSAY_ID = "GFP_AEQVI_Sarkisyan_2016"
ASSAY_FILENAME = f"{ASSAY_ID}.csv"
REFERENCE_FILENAME = "DMS_substitutions.csv"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA = frozenset(AMINO_ACIDS)
SUBSTITUTION = re.compile(f"([{AMINO_ACIDS}])(0|[1-9][0-9]*)([{AMINO_ACIDS}])")
RAW_FIELDS = ("mutant", "mutated_sequence", "DMS_score", "DMS_score_bin")
VARIANT_FIELDS = (
    "variant_id", "mutant", "sequence", "mutation_count", "mutation_positions",
    "wt_amino_acids", "mutant_amino_acids", "fitness", "fitness_bin",
)
MUTATION_FIELDS = (
    "variant_id", "mutant", "mutation_count", "position", "wt_aa",
    "mutant_aa", "substitution",
)
CHECKS = (
    "row_shape", "missing_values", "finite_numeric_values", "binary_label_domain",
    "duplicate_records", "duplicate_mutant", "duplicate_sequence",
    "sequence_length", "standard_amino_acids", "mutation_format",
    "position_bounds", "reference_amino_acid", "duplicate_mutation_positions",
    "amino_acid_changes", "sequence_reconstruction", "label_cutoff_consistency",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{path} needs unique, nonempty column headers")
        return fields, [(reader.line_num, row) for row in reader]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_reference(path):
    _, rows = read_csv(path)
    matches = [row for _, row in rows if row.get("DMS_id") == ASSAY_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected one reference record for {ASSAY_ID}, got {len(matches)}")
    ref = matches[0]
    wt = ref["target_seq"]
    if len(wt) != int(ref["seq_len"]) or set(wt) - STANDARD_AA:
        raise ValueError("Reference sequence has an invalid length or amino acid")
    cutoff = Decimal(ref["DMS_binarization_cutoff"])
    direction = int(ref["raw_DMS_directionality"])
    if not cutoff.is_finite() or direction != 1:
        raise ValueError("Expected a finite GFP cutoff and a positive fitness directionality")
    if ref["DMS_filename"] != ASSAY_FILENAME:
        raise ValueError(f"Unexpected reference assay filename: {ref['DMS_filename']}")
    metadata = {
        "dataset_name": ref["DMS_id"],
        "assay_index": ref["DMS_index"],
        "protein_name": ref["molecule_name"],
        "uniprot_id": ref["UniProt_ID"],
        "source_organism": ref["source_organism"],
        "sequence_length": len(wt),
        "assay_type": ref["selection_type"],
        "selection_assay": ref["selection_assay"],
        "phenotype": ref["raw_DMS_phenotype_name"],
        "fitness_directionality": direction,
        "fitness_directionality_description": "Higher DMS_score indicates higher fluorescence.",
        "binarization_cutoff": float(cutoff),
        "binarization_cutoff_source_value": str(cutoff),
        "binarization_method": ref["DMS_binarization_method"],
        "binary_label_rule": "DMS_score >= cutoff",
        "literature_title": ref["title"],
        "literature_doi": ref["jo"],
        "literature_year": int(ref["year"]),
        "proteingym_download_release": "v1.3",
        "proteingym_reference_row_version": ref["ProteinGym_version"],
        "reference_file_scope": "GFP assay row from the ProteinGym v1.3 reference table",
        "mutated_region_reported_by_source": ref["region_mutated"],
        "reference_total_variants": int(ref["DMS_total_number_mutants"]),
        "reference_single_variants": int(ref["DMS_number_single_mutants"]),
        "reference_multi_variants": int(ref["DMS_number_multiple_mutants"]),
    }
    return wt, cutoff, metadata


def audit_rows(rows, wt, cutoff):
    variants, mutations, wt_rows = [], [], []
    issues = {name: [] for name in CHECKS}
    seen = {check: defaultdict(list) for check in
            ("duplicate_records", "duplicate_mutant", "duplicate_sequence")}

    for index, (line, row) in enumerate(rows, start=1):
        variant_id = f"GFP_{index:06d}"
        mutant, sequence, score, label = (row.get(field) or "" for field in RAW_FIELDS)
        identity = {"source_line": line, "variant_id": variant_id, "mutant": mutant}

        def flag(check, **details):
            issues[check].append({**identity, **details})

        if None in row or any(row.get(field) is None for field in RAW_FIELDS):
            flag("row_shape", reason="Record width differs from header width")
        missing = [field for field in RAW_FIELDS if not (row.get(field) or "").strip()]
        if missing:
            flag("missing_values", fields=missing)

        values = {}
        for field in ("DMS_score", "DMS_score_bin"):
            text = row.get(field) or ""
            if not text.strip():
                continue
            try:
                value = Decimal(text)
            except InvalidOperation:
                flag("finite_numeric_values", field=field, observed=text, reason="Invalid number")
                continue
            if not value.is_finite():
                flag("finite_numeric_values", field=field, observed=text, reason="Non-finite number")
                continue
            values[field] = value

        fitness = values.get("DMS_score")
        binary = values.get("DMS_score_bin")
        if binary is not None and binary not in (0, 1):
            flag("binary_label_domain", observed=label)
        if fitness is not None and binary is not None:
            expected = int(fitness >= cutoff)
            if expected != binary:
                flag("label_cutoff_consistency", fitness=score, cutoff=str(cutoff),
                     observed=label, expected=expected)

        seen["duplicate_records"][(mutant, sequence, score, label)].append(identity)
        if mutant:
            seen["duplicate_mutant"][mutant].append(identity)
        if sequence:
            seen["duplicate_sequence"][sequence].append(identity)
            if len(sequence) != len(wt):
                flag("sequence_length", observed=len(sequence), expected=len(wt))
            invalid = sorted(set(sequence) - STANDARD_AA)
            if invalid:
                flag("standard_amino_acids", invalid_residues=invalid)
        if sequence == wt:
            wt_rows.append({**identity, "fitness": score if fitness is not None else None})

        tokens = [] if mutant in ("WT", "wildtype") else mutant.split(":")
        parsed = []
        for token in tokens:
            match = SUBSTITUTION.fullmatch(token)
            if not match:
                flag("mutation_format", token=token, reason="Expected a standard substitution such as K3R")
                continue
            ref_aa, position, alt_aa = match.groups()
            parsed.append((int(position), ref_aa, alt_aa, token))
        repeated = sorted(p for p, n in Counter(item[0] for item in parsed).items() if n > 1)
        if repeated:
            flag("duplicate_mutation_positions", positions=repeated)

        reconstructed = list(wt)
        for position, ref_aa, alt_aa, token in parsed:
            if ref_aa == alt_aa:
                flag("amino_acid_changes", token=token, reason="Reference and alternate residue are equal")
            if not 1 <= position <= len(wt):
                flag("position_bounds", token=token, position=position, maximum=len(wt))
                continue
            if wt[position - 1] != ref_aa:
                flag("reference_amino_acid", token=token, expected=wt[position - 1], observed=ref_aa)
            reconstructed[position - 1] = alt_aa
        if len(parsed) == len(tokens) and not repeated and all(1 <= p <= len(wt) for p, *_ in parsed):
            rebuilt = "".join(reconstructed)
            if rebuilt != sequence:
                flag("sequence_reconstruction", expected_sequence=rebuilt, observed_sequence=sequence)

        # Keep the score text verbatim to preserve the precision of the source CSV.
        variants.append({
            "variant_id": variant_id, "mutant": mutant, "sequence": sequence,
            "mutation_count": len(parsed),
            "mutation_positions": ";".join(str(p) for p, *_ in parsed),
            "wt_amino_acids": ";".join(ref_aa for _, ref_aa, _, _ in parsed),
            "mutant_amino_acids": ";".join(alt_aa for _, _, alt_aa, _ in parsed),
            "fitness": score, "fitness_bin": int(binary) if binary in (0, 1) else label,
        })
        mutations.extend({
            "variant_id": variant_id, "mutant": mutant, "mutation_count": len(parsed),
            "position": p, "wt_aa": ref_aa, "mutant_aa": alt_aa, "substitution": token,
        } for p, ref_aa, alt_aa, token in parsed)

    for check, groups in seen.items():
        for group in groups.values():
            if len(group) > 1:
                lines = [entry["source_line"] for entry in group]
                issues[check].extend({**entry, "duplicate_group_source_lines": lines} for entry in group)
    return variants, mutations, wt_rows, issues


def process(root):
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    eda = root / "results" / "eda"
    assay_path, ref_path = raw / ASSAY_FILENAME, raw / REFERENCE_FILENAME
    eda.mkdir(parents=True, exist_ok=True)
    report_path = eda / "data_quality_report.json"
    try:
        wt, cutoff, metadata = load_reference(ref_path)
        headers, rows = read_csv(assay_path)
        if tuple(headers) != RAW_FIELDS or not rows:
            raise ValueError(f"Expected a nonempty assay with columns {RAW_FIELDS}")
    except (ValueError, KeyError, TypeError, InvalidOperation, OSError, csv.Error) as error:
        write_json(report_path, {"dataset": ASSAY_ID, "status": "failed", "input_error": str(error)})
        raise
    variants, mutations, wt_rows, issues = audit_rows(rows, wt, cutoff)
    n_single = sum(variant["mutation_count"] == 1 for variant in variants)
    n_multi = sum(variant["mutation_count"] > 1 for variant in variants)
    observed_counts = {"total": len(rows), "single": n_single, "multiple": n_multi}
    reference_counts = {
        "total": metadata["reference_total_variants"], "single": metadata["reference_single_variants"],
        "multiple": metadata["reference_multi_variants"],
    }
    count_match = observed_counts == reference_counts
    measured_wt = [entry for entry in wt_rows if entry["fitness"] is not None]
    failure = any(issues.values()) or not count_match
    report = {
        "dataset": ASSAY_ID,
        "status": "failed" if failure else "passed",
        "raw_variant_count": len(rows),
        "processed_variant_count": None if failure else len(variants),
        "mutation_long_row_count": None if failure else len(mutations),
        "reference_counts": {"passed": count_match, "expected": reference_counts, "observed": observed_counts},
        "source_files_sha256": {path.name: file_sha256(path) for path in (assay_path, ref_path)},
        "checks": {check: {
            "passed": not entries,
            "affected_record_count": len({entry["source_line"] for entry in entries}),
            "records": entries,
        } for check, entries in issues.items()},
        "wild_type": {
            "status": "measured" if measured_wt else "sequence_without_valid_fitness" if wt_rows else "absent",
            "sequence_record_count": len(wt_rows),
            "measured_fitness_record_count": len(measured_wt),
            "records": wt_rows,
        },
    }
    write_json(report_path, report)
    if failure:
        raise ValueError(f"Quality checks failed. See {report_path} for records and reasons.")

    processed.mkdir(parents=True, exist_ok=True)
    (processed / "gfp_wildtype.fasta").write_text(
        f">{ASSAY_ID}|ProteinGym_reference\n{fill(wt, width=70)}\n", encoding="utf-8", newline="\n"
    )
    metadata["source_files_sha256"] = report["source_files_sha256"]
    write_json(processed / "gfp_metadata.json", metadata)
    write_csv(processed / "gfp_processed.csv", VARIANT_FIELDS, variants)
    write_csv(processed / "gfp_mutations_long.csv", MUTATION_FIELDS, mutations)
    print(f"Validated {len(variants):,} variants and wrote {len(mutations):,} substitution rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    process(parser.parse_args().root.resolve())
