"""Block 3 data contracts only. Tiny certification fixtures, no routing sweeps."""
import copy
import json
from pathlib import Path
import tempfile
from time import sleep
import unittest
from unittest.mock import patch

from m3d.checker import check
from m3d.generator import GenConfig, generate_candidate, generate_feasible
from m3d.model import Instance, Submission
from m3d.routing_data import (TrainingSampler, fingerprints, load_config,
                              load_split, plan_cases, requested_config)
from m3d.data_generation import _worker, audit_dataset, generate_dataset


CONFIG = Path(__file__).resolve().parents[1] / "configs/routing_data_v1.json"


def tiny_config():
    config = load_config(CONFIG)
    config["counts"] = {"train": 4, "val": 4, "test": 4}
    config["limits"] = {"total_seconds": 20, "case_seconds": 2,
                        "attempt_seconds": 2, "case_retries": 1}
    for family in config["families"]:
        config["families"][family] = {
            "width": 10, "height": 10, "n_nets": 2, "max_fanout": 3,
            "p_twopin": 0.5, "frac_local": 0.5, "frac_cross": 0.5,
            "cell_min": 1, "cell_max": 1, "pins_per_cell": 1,
            "router": "baseline"}
    return config


class RoutingDataTests(unittest.TestCase):
    def test_fingerprints_ignore_ids_order_metadata_and_geometry_ignores_delay(self):
        inst = generate_candidate(GenConfig(width=14, height=14, n_nets=3,
                                            cell_min=1, cell_max=1, cell_gap=0), 97)
        changed = copy.deepcopy(inst)
        mapping = {pin.id: pin.id + 1000 for pin in changed.pins}
        for pin in changed.pins:
            pin.id = mapping[pin.id]
            pin.cell += 2000
        for cell in changed.cells:
            cell.id += 2000
        for net in changed.nets:
            net.id += 900
            net.driver = mapping[net.driver]
            net.sinks = [mapping[s] for s in reversed(net.sinks)]
        changed.pins.reverse()
        changed.nets.reverse()
        changed.cells.reverse()
        changed.name, changed.seed, changed.params = "renamed", 999, {"timestamp": 123}
        self.assertEqual(fingerprints(inst), fingerprints(changed))
        changed.via_delay += 1
        self.assertEqual(fingerprints(inst)["geometry"], fingerprints(changed)["geometry"])
        self.assertNotEqual(fingerprints(inst)["full_instance"], fingerprints(changed)["full_instance"])

    def test_stable_slots_requested_parameters_and_reproducible_candidates(self):
        config = tiny_config()
        expanded = copy.deepcopy(config)
        expanded["counts"]["train"] = 12
        first = [slot for slot in plan_cases(config) if slot["split"] != "train"]
        second = [slot for slot in plan_cases(expanded) if slot["split"] != "train"]
        self.assertEqual(first, second)
        slot = first[0]
        a = requested_config(config, slot, 0)
        b = requested_config(expanded, slot, 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a.seed, requested_config(config, slot, 1).seed)
        self.assertEqual(fingerprints(generate_candidate(a, a.seed)),
                         fingerprints(generate_candidate(b, b.seed)))

    def test_certification_resume_split_integrity_and_clean_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            config = tiny_config()
            report = generate_dataset(config, root)
            self.assertEqual(report["status"], "complete", report)
            self.assertEqual(report["accepted"], 12)
            manifest = json.loads((root / "manifest.json").read_text())
            original = {e["case_id"]: e for e in manifest["cases"]}
            files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.glob("instances/*/*.json")}
            for entry in manifest["cases"]:
                inst = Instance.load(str(root / entry["instance_file"]))
                cert = Submission.load(str(root / entry["certificate_file"]))
                checked = check(inst, cert)
                self.assertTrue(checked.legal)
                self.assertEqual(checked.total_delay, entry["reference_total_delay"])
            audit = audit_dataset(root)
            self.assertEqual(audit["checked_certificates"], 12)
            self.assertEqual(audit["duplicate_count"], 0)
            expanded = copy.deepcopy(config)
            expanded["counts"]["train"] = 8
            self.assertEqual(generate_dataset(expanded, root)["accepted"], 16)
            updated = json.loads((root / "manifest.json").read_text())
            for entry in updated["cases"]:
                if entry["case_id"] in original:
                    self.assertEqual(entry, original[entry["case_id"]])
            for path, before in files.items():
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            # A clean run with more train slots has identical val/test membership.
            other = Path(directory) / "other"
            generate_dataset(expanded, other)
            other_manifest = json.loads((other / "manifest.json").read_text())
            self.assertEqual({e["case_id"]: e["fingerprints"] for e in manifest["cases"] if e["split"] != "train"},
                             {e["case_id"]: e["fingerprints"] for e in other_manifest["cases"] if e["split"] != "train"})
            # Certificates can be absent: the loader must never open them.
            (root / "certificates").rename(root / "hidden-certificates")
            a, b = TrainingSampler(root, seed=8), TrainingSampler(root, seed=8)
            sequence_a = [a.sample()[1] for _ in range(12)]
            sequence_b = [b.sample()[1] for _ in range(12)]
            self.assertEqual(sequence_a, sequence_b)
            self.assertTrue(all(m.split == "train" for m in sequence_a))
            inst, metadata = load_split(root, split="val").load(0)
            self.assertEqual(metadata.split, "val")
            self.assertNotIn("routes", inst.to_dict())
            self.assertFalse(hasattr(metadata, "certificate_file"))
            with self.assertRaises(TypeError):
                load_split(root)

    def test_resume_rejects_parameter_changes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config()
            generate_dataset(config, root)
            changed = copy.deepcopy(config)
            changed["defaults"]["via_delay"] += 1
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate_dataset(changed, root)
            manifest = json.loads((root / "manifest.json").read_text())
            target = root / manifest["cases"][0]["instance_file"]
            target.write_text(target.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash"):
                audit_dataset(root)

    def test_duplicate_rejection_partial_status_and_requested_effective_growth(self):
        config = tiny_config()
        result = generate_feasible(requested_config(config, plan_cases(config)[0], 0))
        # Simulate an upstream enlarged grid without changing any routing
        # connectivity. Its existing certificate remains checker-legal.
        result.instance.width += 6
        result.instance.params["width"] += 6

        def same_geometry(cfg, directory, timeout_s):
            instance, certificate = copy.deepcopy(result.instance), copy.deepcopy(result.reference)
            instance.name = certificate.instance = cfg.name
            return {"status": "certified", "instance": instance.to_dict(),
                    "certificate": certificate.to_dict(), "reference_total_delay": result.baseline_total,
                    "checker_legal": True, "elapsed_s": 0.01, "timings": {},
                    "events": [{"event": "candidate_started"}]}

        with tempfile.TemporaryDirectory() as directory, patch("m3d.data_generation._worker", same_geometry):
            root = Path(directory)
            report = generate_dataset(config, root)
            self.assertEqual((report["status"], report["accepted"], report["rejected_attempts"]), ("partial", 1, 11))
            entry = json.loads((root / "manifest.json").read_text())["cases"][0]
            self.assertEqual(entry["requested_parameters"]["width"], 10)
            self.assertEqual(entry["effective"]["width"], 16)
            self.assertTrue(entry["effective"]["grid_enlarged"])
            self.assertLess(entry["effective"]["net_density_per_xy_vertex"],
                            entry["effective"]["requested_net_density_per_xy_vertex"])
            self.assertTrue(all(a["status"] == "duplicate" for a in report["attempts"][1:]))
            # Resume must not silently give exhausted slots more retry budget.
            resumed = generate_dataset(config, root)
            self.assertEqual(len(resumed["attempts"]), 12)
        exclusion = {"inputs": [{"geometry": fingerprints(result.instance)["geometry"]}],
                     "count": 1, "catalog_sha256": "test-catalog"}
        with tempfile.TemporaryDirectory() as directory, patch("m3d.data_generation._worker", same_geometry), \
                patch("m3d.data_generation.released_inputs", return_value=exclusion):
            report = generate_dataset(config, Path(directory))
            self.assertEqual((report["accepted"], report["rejected_attempts"]), (0, 12))
            self.assertTrue(all(a["reason"] == "released geometry" for a in report["attempts"]))

    def test_worker_timeout_is_not_infeasibility_and_total_cap_is_partial(self):
        cfg = requested_config(tiny_config(), plan_cases(tiny_config())[0], 0)
        with tempfile.TemporaryDirectory() as directory:
            result = _worker(cfg, Path(directory) / "worker", 0.000001)
            self.assertEqual(result["status"], "timeout")
            self.assertIn("not proof of infeasibility", result["reason"])
            config = tiny_config()
            config["limits"]["total_seconds"] = 0.000001
            report = generate_dataset(config, Path(directory) / "capped")
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(report["attempts"], [])
            self.assertEqual(len(report["shortfalls"]), 12)

    def test_manifest_cannot_silently_move_held_out_case_to_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_dataset(tiny_config(), root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["cases"][0]["split"] = "train"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "split|slot"):
                audit_dataset(root)
            with self.assertRaisesRegex(ValueError, "split|slot"):
                TrainingSampler(root, seed=1)

    def test_late_certification_is_not_published_and_audit_can_be_pending(self):
        config = tiny_config()
        cfg = requested_config(config, plan_cases(config)[0], 0)
        result = generate_feasible(cfg)
        config["limits"]["total_seconds"] = 0.5
        config["limits"]["case_seconds"] = 0.1

        def late_worker(*_):
            sleep(0.2)
            return {"status": "certified", "instance": result.instance.to_dict(),
                    "certificate": result.reference.to_dict(), "reference_total_delay": result.baseline_total,
                    "checker_legal": True, "elapsed_s": 0.2, "timings": {}, "events": []}

        with tempfile.TemporaryDirectory() as directory, patch("m3d.data_generation._worker", late_worker):
            report = generate_dataset(config, Path(directory))
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(report["status"], "partial")
            self.assertTrue(all(a["status"] == "timeout" for a in report["attempts"]))


if __name__ == "__main__":
    unittest.main()
