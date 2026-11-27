"""Sparse, position-major protein sequence features."""

import hashlib
import json

import numpy as np
from scipy.sparse import csr_matrix

from data_processing import AMINO_ACIDS, ASSAY_ID, read_csv

SPLIT_SIZES = {"train_initial": 3000, "validation": 1000, "test": 5000, "candidate_pool": 42714}
ID = "experiment_variant_id"
SEQUENCE_FIELDS = (ID, "mutant", "sequence", "mutation_count", "mutation_positions",
                   "wt_amino_acids", "mutant_amino_acids")
LABELED_FIELDS = (*SEQUENCE_FIELDS, "fitness", "fitness_bin")
REFERENCE_INPUTS = ("data/processed/gfp_wildtype.fasta", "data/processed/gfp_metadata.json")

_LOOKUP = np.full(256, -1, dtype=np.int16)
_LOOKUP[np.frombuffer(AMINO_ACIDS.encode("ascii"), dtype=np.uint8)] = np.arange(20)


class SequenceEncoder:
    def __init__(self, reference):
        if not isinstance(reference, str) or not reference:
            raise ValueError("Reference sequence must be a nonempty string")
        self.reference = reference
        self.length = len(reference)
        self.reference_codes = self._codes([reference])[0]

    def _codes(self, sequences):
        rows = []
        for i, sequence in enumerate(sequences):
            if not isinstance(sequence, str) or len(sequence) != self.length:
                raise ValueError(f"Sequence {i} must have length {self.length}")
            try:
                codes = _LOOKUP[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
            except UnicodeEncodeError as error:
                raise ValueError(f"Invalid amino acid in sequence {i}") from error
            invalid = np.flatnonzero(codes < 0)
            if invalid.size:
                pos = int(invalid[0])
                raise ValueError(f"Invalid amino acid {sequence[pos]!r} in sequence {i} at position {pos + 1}")
            rows.append(codes)
        return np.asarray(rows, dtype=np.int16).reshape(-1, self.length)

    def mutation_counts(self, sequences):
        batch = [sequences] if isinstance(sequences, str) else sequences
        return np.count_nonzero(self._codes(batch) != self.reference_codes, axis=1)

    def transform(self, sequences, include_count=False):
        batch = [sequences] if isinstance(sequences, str) else sequences
        codes = self._codes(batch)
        width = self.length * len(AMINO_ACIDS)
        indices = codes + np.arange(self.length, dtype=np.int32) * len(AMINO_ACIDS)
        values = np.ones(codes.shape, dtype=np.float32)
        if include_count:
            counts = np.count_nonzero(codes != self.reference_codes, axis=1)
            indices = np.column_stack((indices, np.full(len(codes), width, dtype=np.int32)))
            values = np.column_stack((values, counts.astype(np.float32)))
        step = self.length + int(include_count)
        pointers = np.arange(0, (len(codes) + 1) * step, step, dtype=np.int32)
        return csr_matrix((values.ravel(), indices.ravel(), pointers),
                          shape=(len(codes), width + int(include_count)), dtype=np.float32)


def load_reference(root):
    fasta, metadata = (root / name for name in REFERENCE_INPUTS)
    lines = fasta.read_text(encoding="utf-8").splitlines()
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if (not lines or not lines[0].startswith(f">{ASSAY_ID}|")
            or meta.get("dataset_name") != ASSAY_ID or meta.get("sequence_length") != 238):
        raise ValueError("Expected the 238-residue GFP reference and matching assay metadata")
    reference = "".join(lines[1:])
    if len(reference) != meta["sequence_length"]:
        raise ValueError("Reference FASTA length differs from assay metadata")
    cutoff = float(meta["binarization_cutoff_source_value"])
    if not np.isfinite(cutoff):
        raise ValueError("Invalid assay fitness threshold")
    return SequenceEncoder(reference), cutoff


def load_split(root, name, encoder):
    expected = SPLIT_SIZES[name]
    unlabeled = name == "candidate_pool"
    fields = SEQUENCE_FIELDS if unlabeled else LABELED_FIELDS
    header, source = read_csv(root / "data" / "splits" / f"{name}.csv")
    if tuple(header) != fields or len(source) != expected:
        raise ValueError(f"{name}: expected the Stage 2 schema and {expected:,} records")
    rows = [row for _, row in source]
    if any(set(row) != set(fields) or any(value is None or not value.strip() for value in row.values())
           for row in rows):
        raise ValueError(f"{name}: malformed or missing CSV fields")
    sequences = [row["sequence"] for row in rows]
    counts = encoder.mutation_counts(sequences)
    identifiers = set()
    for (_, row), count in zip(source, counts):
        identifier = hashlib.sha256(row["sequence"].encode("ascii")).hexdigest()
        if identifier != row[ID] or identifier in identifiers:
            raise ValueError(f"{name}: invalid or duplicate sequence ID {row[ID]}")
        identifiers.add(identifier)
        if int(row["mutation_count"]) != count or not 1 <= count <= 15:
            raise ValueError(f"{name}: incorrect mutation count for {identifier}")
        if not unlabeled and (not np.isfinite(float(row["fitness"])) or row["fitness_bin"] not in {"0", "1"}):
            raise ValueError(f"{name}: invalid fitness or binary label for {identifier}")
    return rows


def check_disjoint(splits):
    ids, sequences = set(), set()
    for name, rows in splits.items():
        new_ids = {row[ID] for row in rows}
        new_sequences = {row["sequence"] for row in rows}
        if len(new_ids) != len(rows) or len(new_sequences) != len(rows):
            raise ValueError(f"Duplicate sequences or IDs in {name}")
        if ids & new_ids or sequences & new_sequences:
            raise ValueError(f"Overlapping sequences or IDs in {name}")
        ids.update(new_ids)
        sequences.update(new_sequences)
