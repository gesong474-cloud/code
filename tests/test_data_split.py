import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_split import (
    CANDIDATE_FIELDS, GROUPS, ID, LABELED_FIELDS, MANIFEST_FIELDS, ORACLE_FIELDS, SPLITS,
    experiment_variant_id, index_variants, load_export, make_splits, stratification_group,
    validate_exports, validate_reproducibility, write_split_tables,
)
from data_processing import write_csv


class DataSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Distinct sequences with diverse substitution counts and deliberately ordered fitness.
        wt = "A" * 30
        counts = (1, 2, 3, 4, 5, 6, 15)
        alph = "CDEFGHIKLMNPQRSTVWY"
        rows = []
        for count in counts:
            for replicate in range(30):
                start = replicate % 15
                residues = alph[replicate // 15] + "C" * (count - 1)
                sequence = wt[:start] + residues + wt[start + count:]
                positions = list(range(start + 1, start + count + 1))
                mutants = [f"A{p}{aa}" for p, aa in zip(positions, residues)]
                rows.append((len(rows) + 2, {
                    "variant_id": f"GFP_{len(rows) + 1:06d}", "mutant": ":".join(mutants),
                    "sequence": sequence, "mutation_count": str(count),
                    "mutation_positions": ";".join(map(str, positions)),
                    "wt_amino_acids": ";".join("A" * count), "mutant_amino_acids": ";".join(residues),
                    "fitness": "1.50100000000001" if replicate < 9 else "3.60200000000001",
                    "fitness_bin": "0" if replicate < 9 else "1",
                }))
        cls.wt, cls.rows = wt, rows
        cls.variants = index_variants(rows, wt)
        cls.groups = {key: row["stratification_group"] for key, row in cls.variants.items()}
        cls.sizes = dict(zip(SPLITS, (35, 21, 28, 126)))
        cls.partition = make_splits(cls.groups, cls.sizes, 42)

    def test_group_boundaries(self):
        self.assertEqual([stratification_group(n) for n in range(1, 16)],
                         [*GROUPS[:5], "6-7", "6-7", *(["8-15"] * 8)])
        for invalid in (0, 16):
            with self.assertRaisesRegex(ValueError, "outside"):
                stratification_group(invalid)

    def test_hash_id_is_stable_and_unique(self):
        sequence = self.rows[0][1]["sequence"]
        self.assertEqual(experiment_variant_id(sequence), hashlib.sha256(sequence.encode("ascii")).hexdigest())
        self.assertEqual(len(self.variants), len(self.rows))
        self.assertEqual(set(index_variants(list(reversed(self.rows)), self.wt)), set(self.variants))

    def test_duplicate_sequence_and_hash_collision_rejected(self):
        duplicate = {**self.rows[0][1], "variant_id": "extra-source-id"}
        with self.assertRaisesRegex(ValueError, "duplicate sequence"):
            index_variants([self.rows[0], (1000, duplicate)], self.wt)
        with patch("data_split.experiment_variant_id", return_value="synthetic-collision"):
            with self.assertRaisesRegex(ValueError, "SHA-256 collision"):
                index_variants(self.rows[:2], self.wt)

    def test_counts_disjointness_full_coverage_and_groups(self):
        all_ids, all_sequences = set(), set()
        for name, keys in self.partition.items():
            sequences = {self.variants[key]["sequence"] for key in keys}
            self.assertEqual(len(keys), self.sizes[name])
            self.assertEqual(len(sequences), len(keys))
            self.assertFalse(all_ids.intersection(keys))
            self.assertFalse(all_sequences.intersection(sequences))
            self.assertEqual({self.groups[key] for key in keys}, set(GROUPS))
            all_ids.update(keys)
            all_sequences.update(sequences)
        self.assertEqual(all_ids, set(self.variants))
        self.assertEqual(all_sequences, {row["sequence"] for row in self.variants.values()})

    def test_same_and_different_seed(self):
        report = validate_reproducibility(self.groups, self.partition, self.sizes, 42)
        self.assertTrue(report["same_seed_identical_assignment_and_order"])
        self.assertTrue(report["different_seed_distinct_valid_partition"])
        self.assertGreater(report["membership_changes_by_split"]["train_initial"], 0)

    def test_allocation_is_independent_of_labels_and_input_order(self):
        changed = [(line, {**row, "fitness": "2.5", "fitness_bin": "1"}) for line, row in reversed(self.rows)]
        remapped = index_variants(changed, self.wt)
        groups = {key: row["stratification_group"] for key, row in remapped.items()}
        self.assertEqual(make_splits(groups, self.sizes, 42), self.partition)

    def test_wrong_target_total_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to"):
            make_splits(self.groups, {**self.sizes, "train_initial": 34}, 42)

    def test_outputs_hide_labels_and_preserve_exact_fitness(self):
        with TemporaryDirectory(prefix=".gfp-split-test-", dir=ROOT) as temporary:
            path = Path(temporary)
            write_split_tables(path, self.variants, self.partition, 42)
            self.assertTrue(all(validate_exports(path, self.variants, self.partition, self.sizes).values()))
            pool = load_export(path / "candidate_pool.csv", CANDIDATE_FIELDS)
            labeled = load_export(path / "train_initial.csv", LABELED_FIELDS)
            oracle = load_export(path / "oracle_labels.csv", ORACLE_FIELDS)
            manifest = load_export(path / "split_manifest.csv", MANIFEST_FIELDS)
            for field in ("fitness", "fitness_bin", "variant_id", "original_variant_id"):
                self.assertNotIn(field, pool[0])
            self.assertNotIn("variant_id", labeled[0])
            self.assertEqual({row[ID] for row in pool}, {row[ID] for row in oracle})
            self.assertNotEqual([row[ID] for row in pool], [row[ID] for row in oracle])
            self.assertEqual({row["original_variant_id"] for row in manifest},
                             {row["variant_id"] for row in self.variants.values()})
            for row in oracle + labeled:
                self.assertEqual(row["fitness"], self.variants[row[ID]]["fitness"])

    def test_candidate_label_leak_is_rejected(self):
        with TemporaryDirectory(prefix=".gfp-split-test-", dir=ROOT) as temporary:
            path = Path(temporary)
            write_split_tables(path, self.variants, self.partition, 42)
            pool = load_export(path / "candidate_pool.csv", CANDIDATE_FIELDS)
            write_csv(path / "candidate_pool.csv", (*CANDIDATE_FIELDS, "fitness"),
                      ({**row, "fitness": self.variants[row[ID]]["fitness"]} for row in pool))
            with self.assertRaisesRegex(ValueError, "candidate_pool.csv has unexpected columns"):
                validate_exports(path, self.variants, self.partition, self.sizes)

    def test_oracle_duplicate_or_mismatch_is_rejected(self):
        with TemporaryDirectory(prefix=".gfp-split-test-", dir=ROOT) as temporary:
            path = Path(temporary)
            write_split_tables(path, self.variants, self.partition, 42)
            oracle = load_export(path / "oracle_labels.csv", ORACLE_FIELDS)
            write_csv(path / "oracle_labels.csv", ORACLE_FIELDS, [oracle[1], *oracle[1:]])
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                validate_exports(path, self.variants, self.partition, self.sizes)
            oracle[0]["fitness"] = "0"
            write_csv(path / "oracle_labels.csv", ORACLE_FIELDS, oracle)
            with self.assertRaisesRegex(ValueError, "differs from source"):
                validate_exports(path, self.variants, self.partition, self.sizes)


if __name__ == "__main__":
    unittest.main()
