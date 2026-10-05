"""Block 3: deterministic case plans, routing fingerprints, and input-only loaders."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from .generator import GenConfig
from .model import Instance

FAMILIES = ("sparse", "mixed", "dense", "branching")
SPLITS = ("train", "val", "test")
REPO = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(instance: Instance) -> dict:
    pins = instance.pin_vertex()
    nets = sorted((pins[n.driver], tuple(sorted(pins[p] for p in n.sinks))) for n in instance.nets)
    geometry = {"grid": [instance.width, instance.height, instance.layers], "nets": nets}
    return {"geometry": digest(geometry),
            "full_instance": digest({**geometry, "layer_delay": instance.layer_delay,
                                     "via_delay": instance.via_delay})}


def derive_seed(master: int, split: str, family: str, index: int, retry: int,
                purpose: str = "generation") -> int:
    return int(digest(["routing-data-v1", master, split, family, index, retry, purpose])[:16], 16)


def validate_config(config: dict) -> dict:
    if set(config) != {"version", "master_seed", "counts", "limits", "defaults", "families"}:
        raise ValueError("configuration has missing or unknown fields")
    if config["version"] != 1 or type(config["master_seed"]) is not int or config["master_seed"] < 0:
        raise ValueError("expected version 1 and nonnegative integer master_seed")
    if set(config["counts"]) != set(SPLITS):
        raise ValueError("counts must explicitly name train, val, and test")
    for count in config["counts"].values():
        if type(count) is not int or count < 4 or count % 4:
            raise ValueError("each split count must be a positive multiple of four")
    limits = config["limits"]
    if set(limits) != {"total_seconds", "case_seconds", "attempt_seconds", "case_retries"}:
        raise ValueError("unknown or missing generation limits")
    for key, value in limits.items():
        if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                or (key == "case_retries" and type(value) is not int)):
            raise ValueError(f"invalid finite limit: {key}")
    if set(config["families"]) != set(FAMILIES):
        raise ValueError("expected sparse, mixed, dense, and branching families")
    defaults = asdict(GenConfig())
    allowed = defaults.keys() - {"name", "seed", "master_seed"}
    for overrides in [config["defaults"], *config["families"].values()]:
        if not overrides.keys() <= allowed:
            raise ValueError("unsupported GenConfig parameter")
        for key, specification in overrides.items():
            values = specification if isinstance(specification, list) else [specification]
            if isinstance(specification, list) and (len(values) != 2 or values[0] > values[1]):
                raise ValueError(f"invalid inclusive range: {key}")
            for value in values:
                if type(defaults[key]) is int:
                    minimum = 0 if key in ("layer_slope", "cell_gap") else 2 if key in ("layers", "max_fanout") else 1
                    if type(value) is not int or value < minimum:
                        raise ValueError(f"invalid integer parameter: {key}")
                elif type(defaults[key]) is float:
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"invalid fraction: {key}")
                elif isinstance(specification, list) or value not in (
                        ("baseline", "negotiated") if key == "router" else ("bbox_desc", "bbox_asc", "id")):
                    raise ValueError(f"invalid choice: {key}")
    for family in FAMILIES:
        merged = {**defaults, **config["defaults"], **config["families"][family]}
        lo = merged["cell_min"]
        hi = merged["cell_max"]
        if (max(lo) if isinstance(lo, list) else lo) > (min(hi) if isinstance(hi, list) else hi):
            raise ValueError("cell_min range can exceed cell_max")
    return config


def load_config(path) -> dict:
    return validate_config(json.loads(Path(path).read_text()))


def plan_cases(config: dict) -> list[dict]:
    validate_config(config)
    # Reserve held-out slots first. Training growth cannot win a duplicate
    # collision against validation/test, even in a fresh larger dataset.
    return [{"case_id": f"{split}-{family}-{index:05d}", "split": split,
             "family": family, "index": index}
            for split in ("test", "val", "train")
            for index in range(config["counts"][split] // len(FAMILIES))
            for family in FAMILIES]


def requested_config(config: dict, slot: dict, retry: int) -> GenConfig:
    args = (config["master_seed"], slot["split"], slot["family"], slot["index"])
    rng = random.Random(derive_seed(*args, 0, "parameters"))
    parameters = {**config["defaults"], **config["families"][slot["family"]]}
    for key in sorted(parameters):
        value = parameters[key]
        if isinstance(value, list):
            parameters[key] = rng.randint(*value) if type(asdict(GenConfig())[key]) is int else rng.uniform(*value)
    return GenConfig(**parameters, name=slot["case_id"], seed=derive_seed(*args, retry),
                     master_seed=config["master_seed"])


def dataset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("dataset path escapes its root")
    return path


def validate_entry_slot(entry: dict, plan: dict):
    expected = plan.get(entry["case_id"])
    if expected is None or any(entry[key] != expected[key] for key in ("split", "family", "index")):
        raise ValueError(f"entry does not match its fixed split slot: {entry['case_id']}")
    if (entry["instance_file"] != f"instances/{entry['split']}/{entry['case_id']}.json"
            or entry["certificate_file"] != f"certificates/{entry['split']}/{entry['case_id']}.sol.json"):
        raise ValueError("file location does not match fixed split slot")


@dataclass(frozen=True)
class ReferenceMetadata:
    case_id: str
    split: str
    family: str
    reference_total_delay: int
    geometry_hash: str
    full_instance_hash: str


class RoutingSplit:
    def __init__(self, root, *, split: str):
        if split not in SPLITS:
            raise ValueError("explicit split must be train, val, or test")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != "m3d-routing-data-v1":
            raise ValueError("unknown dataset format")
        self.status = manifest["status"]
        self.entries = tuple(sorted((e for e in manifest["cases"] if e["split"] == split),
                                    key=lambda e: e["case_id"]))
        plan = {slot["case_id"]: slot for slot in plan_cases(manifest["configuration"])}
        for entry in self.entries:
            validate_entry_slot(entry, plan)

    def __len__(self):
        return len(self.entries)

    def load(self, index: int) -> tuple[Instance, ReferenceMetadata]:
        entry = self.entries[index]
        path = dataset_path(self.root, entry["instance_file"])
        if file_hash(path) != entry["instance_file_sha256"]:
            raise ValueError(f"instance file hash mismatch: {entry['case_id']}")
        raw = json.loads(path.read_text())
        if "routes" in raw:
            raise ValueError("routing inputs must not contain reference wires")
        instance = Instance.from_dict(raw)
        if fingerprints(instance) != entry["fingerprints"]:
            raise ValueError("routing fingerprint mismatch")
        # Only scalar normalization/provenance leaves this loader. Certificates
        # are deliberately neither opened nor returned.
        return instance, ReferenceMetadata(entry["case_id"], entry["split"], entry["family"],
                                            entry["reference_total_delay"], entry["fingerprints"]["geometry"],
                                            entry["fingerprints"]["full_instance"])


def load_split(root, *, split: str) -> RoutingSplit:
    return RoutingSplit(root, split=split)


class TrainingSampler:
    def __init__(self, root, *, seed: int):
        self._split = load_split(root, split="train")
        if not len(self._split):
            raise ValueError("dataset has no accepted training instances")
        self._rng = random.Random(seed)

    def sample(self) -> tuple[Instance, ReferenceMetadata]:
        return self._split.load(self._rng.randrange(len(self._split)))
