# Copyright (c) 2026, Jiun-Cheng Jiang and the QKAN contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Named launch policies for the flash PZ backward kernel.

With no policy named (``pz_launch_policy=None`` or ``"default"``) the solver
launches the kernel as it always has: ``_select_block_b`` picks the batch tile
and the Triton backend's defaults pick everything else.  Nothing in this
module runs on that path, and it needs no data files.

A named policy is a JSON document, ``flash/policies/<name>.json``, that is
strictly validated and cached.  It records a launch measured to be faster on
one architecture for one PZ shape (``n_oi = out_dim * in_dim`` and the batch)
and mode (``reps``, ``preacts_trainable``, ``fast_measure``)::

    QKAN(
        ...,
        solver="flash",
        solver_kwargs={
            "pz_launch_policy": "pz-gfx942-n256-batch4096-r3-fixed-preacts-fast-v1"
        },
    )

The policy is resolved in the forward pass and applied to the backward launch.
QKAN hands one ``solver_kwargs`` dict to every layer, so a policy meets calls
it was not tuned for: other layers, the short last batch of an epoch, dtypes
it has no entry for.  Those calls keep the default launch, and the first one
warns.  A policy for another architecture is a configuration error and raises
``LaunchPolicyMismatchError`` on the first forward that will run a backward.

Architectures are named ``gfx942`` style on ROCm and ``sm<major><minor>`` on
CUDA (``sm120`` for an RTX 5090).  To add a tuning, run
``scripts/tune_pz_launch_policy.py`` on an otherwise idle GPU of that
architecture and add the JSON document it prints.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from qkan._kernel_utils import _select_block_b

DEFAULT_POLICY_NAME = "default"
_POLICY_PACKAGE = "qkan.solver.flash.policies"
_POLICY_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_PRECISION_NAMES = frozenset(("fp32", "bf16", "fp8"))


class PrecisionKind(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"


class MatchKind(str, Enum):
    ANY = "any"
    EXACT = "exact"


class BlockSelectorKind(str, Enum):
    ORIGINAL = "original"
    FIXED = "fixed"
    PER_PRECISION = "per_precision"


@dataclass(frozen=True)
class ShapeSelection:
    baseline_block: int
    selected_block: int
    baseline_padded_rows: int
    selected_padded_rows: int
    n_b_blocks: int
    n_programs: int
    promoted: bool


@dataclass(frozen=True)
class LaunchConfig:
    block_b: int
    num_warps: int
    waves_per_eu: int
    num_stages: int


@dataclass(frozen=True)
class LaunchMetadata:
    block_b: int | None
    num_warps: int
    waves_per_eu: int
    num_stages: int


@dataclass(frozen=True)
class ArchitectureMatch:
    kind: MatchKind
    value: str | None


@dataclass(frozen=True)
class PolicyMatch:
    kind: MatchKind
    n_oi: int | None
    batch: int | None
    reps: int | None
    preacts_trainable: bool | None
    fast_measure: bool | None


@dataclass(frozen=True)
class BlockSelector:
    kind: BlockSelectorKind
    block_b: int | None


@dataclass(frozen=True)
class LoadedLaunchPolicy:
    schema_version: int
    name: str
    operation: str
    architecture: ArchitectureMatch
    match: PolicyMatch
    block_selector: BlockSelector
    launch_metadata: Mapping[PrecisionKind, LaunchMetadata]
    description: str | None


@dataclass(frozen=True)
class PzBackwardPolicy:
    architecture: str
    precision: PrecisionKind
    selection: ShapeSelection
    launch: LaunchConfig
    policy_name: str


class LaunchPolicyError(ValueError):
    """Base class for explicit launch-policy failures."""


class LaunchPolicyNotFoundError(LaunchPolicyError):
    pass


class LaunchPolicyValidationError(LaunchPolicyError):
    pass


class LaunchPolicyMismatchError(LaunchPolicyError):
    pass


_ROOT_REQUIRED_KEYS = frozenset(
    (
        "schema_version",
        "name",
        "operation",
        "architecture",
        "match",
        "block_selector",
        "precisions",
    )
)
_ROOT_OPTIONAL_KEYS = frozenset(("description",))
_ARCHITECTURE_EXACT_KEYS = frozenset(("kind", "value"))
_MATCH_EXACT_KEYS = frozenset(
    ("kind", "n_oi", "batch", "reps", "preacts_trainable", "fast_measure")
)
_BLOCK_FIXED_KEYS = frozenset(("kind", "block_b"))
_LAUNCH_METADATA_KEYS = frozenset(("num_warps", "waves_per_eu", "num_stages"))
_LAUNCH_METADATA_WITH_BLOCK_KEYS = frozenset(
    ("block_b", "num_warps", "waves_per_eu", "num_stages")
)


def _policy_error(policy_name: str, message: str) -> LaunchPolicyValidationError:
    return LaunchPolicyValidationError(
        f"launch policy {policy_name!r} is invalid: {message}"
    )


def _require_object(value: object, policy_name: str, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _policy_error(policy_name, f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    policy_name: str,
    context: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise _policy_error(policy_name, f"{context} is missing keys {missing}")
    if unknown:
        raise _policy_error(policy_name, f"{context} has unknown keys {unknown}")


def _require_string(value: object, policy_name: str, context: str) -> str:
    if type(value) is not str or not value:
        raise _policy_error(policy_name, f"{context} must be a non-empty string")
    return value


def _require_int(value: object, policy_name: str, context: str) -> int:
    if type(value) is not int:
        raise _policy_error(policy_name, f"{context} must be an integer")
    return value


def _require_bool(value: object, policy_name: str, context: str) -> bool:
    if type(value) is not bool:
        raise _policy_error(policy_name, f"{context} must be a boolean")
    return value


def _validate_policy_name(policy_name: object) -> str:
    if type(policy_name) is not str or not _POLICY_NAME_PATTERN.fullmatch(policy_name):
        raise LaunchPolicyValidationError(
            "policy name must match ^[a-z0-9][a-z0-9_-]{0,127}$"
        )
    return policy_name


def normalize_policy_name(policy_name: str | None) -> str:
    """Map the omitted public value to ``"default"``, the built-in launch."""

    if policy_name is None:
        return DEFAULT_POLICY_NAME
    return _validate_policy_name(policy_name)


def _parse_launch_metadata(
    policy_name: str,
    precision_name: str,
    value: object,
    *,
    requires_block: bool,
) -> LaunchMetadata:
    context = f"precisions.{precision_name}"
    config = _require_object(value, policy_name, context)
    _require_exact_keys(
        config,
        required=(
            _LAUNCH_METADATA_WITH_BLOCK_KEYS
            if requires_block
            else _LAUNCH_METADATA_KEYS
        ),
        policy_name=policy_name,
        context=context,
    )
    num_warps = _require_int(config["num_warps"], policy_name, f"{context}.num_warps")
    waves_per_eu = _require_int(
        config["waves_per_eu"], policy_name, f"{context}.waves_per_eu"
    )
    num_stages = _require_int(
        config["num_stages"], policy_name, f"{context}.num_stages"
    )
    block_b = None
    if requires_block:
        block_b = _require_int(config["block_b"], policy_name, f"{context}.block_b")
        if block_b <= 0 or block_b > 65536 or block_b & (block_b - 1):
            raise _policy_error(
                policy_name,
                f"{context}.block_b must be a power of two from 1 to 65536",
            )
    if num_warps <= 0 or num_warps > 32 or num_warps & (num_warps - 1):
        raise _policy_error(
            policy_name,
            f"{context}.num_warps must be a power of two from 1 to 32",
        )
    if not 0 <= waves_per_eu <= 16:
        raise _policy_error(
            policy_name,
            f"{context}.waves_per_eu must be from 0 to 16",
        )
    if not 1 <= num_stages <= 16:
        raise _policy_error(
            policy_name,
            f"{context}.num_stages must be from 1 to 16",
        )
    return LaunchMetadata(block_b, num_warps, waves_per_eu, num_stages)


def _parse_match_kind(
    value: object,
    policy_name: str,
    context: str,
) -> MatchKind:
    raw_kind = _require_string(value, policy_name, f"{context}.kind")
    try:
        return MatchKind(raw_kind)
    except ValueError as error:
        raise _policy_error(
            policy_name,
            f"{context}.kind must be 'any' or 'exact'",
        ) from error


def _parse_architecture_match(
    policy_name: str,
    value: object,
) -> ArchitectureMatch:
    context = "architecture"
    document = _require_object(value, policy_name, context)
    kind = _parse_match_kind(document.get("kind"), policy_name, context)
    required = (
        frozenset(("kind",)) if kind is MatchKind.ANY else _ARCHITECTURE_EXACT_KEYS
    )
    _require_exact_keys(
        document,
        required=required,
        policy_name=policy_name,
        context=context,
    )
    if kind is MatchKind.ANY:
        return ArchitectureMatch(kind=kind, value=None)
    architecture = _require_string(document["value"], policy_name, "architecture.value")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", architecture):
        raise _policy_error(
            policy_name,
            "architecture.value contains unsupported characters",
        )
    return ArchitectureMatch(kind=kind, value=architecture)


def _parse_policy_match(policy_name: str, value: object) -> PolicyMatch:
    context = "match"
    document = _require_object(value, policy_name, context)
    kind = _parse_match_kind(document.get("kind"), policy_name, context)
    required = frozenset(("kind",)) if kind is MatchKind.ANY else _MATCH_EXACT_KEYS
    _require_exact_keys(
        document,
        required=required,
        policy_name=policy_name,
        context=context,
    )
    if kind is MatchKind.ANY:
        return PolicyMatch(kind, None, None, None, None, None)
    match = PolicyMatch(
        kind=kind,
        n_oi=_require_int(document["n_oi"], policy_name, "match.n_oi"),
        batch=_require_int(document["batch"], policy_name, "match.batch"),
        reps=_require_int(document["reps"], policy_name, "match.reps"),
        preacts_trainable=_require_bool(
            document["preacts_trainable"],
            policy_name,
            "match.preacts_trainable",
        ),
        fast_measure=_require_bool(
            document["fast_measure"],
            policy_name,
            "match.fast_measure",
        ),
    )
    if (
        match.n_oi is None
        or match.batch is None
        or match.reps is None
        or match.n_oi <= 0
        or match.batch <= 0
        or match.reps <= 0
    ):
        raise _policy_error(
            policy_name,
            "match.n_oi, match.batch, and match.reps must be positive",
        )
    return match


def _parse_block_selector(
    policy_name: str,
    value: object,
) -> BlockSelector:
    context = "block_selector"
    document = _require_object(value, policy_name, context)
    raw_kind = _require_string(document.get("kind"), policy_name, f"{context}.kind")
    try:
        kind = BlockSelectorKind(raw_kind)
    except ValueError as error:
        raise _policy_error(
            policy_name,
            "block_selector.kind must be 'original', 'fixed', or 'per_precision'",
        ) from error
    required = (
        frozenset(("kind",))
        if kind
        in (
            BlockSelectorKind.ORIGINAL,
            BlockSelectorKind.PER_PRECISION,
        )
        else _BLOCK_FIXED_KEYS
    )
    _require_exact_keys(
        document,
        required=required,
        policy_name=policy_name,
        context=context,
    )
    if kind in (
        BlockSelectorKind.ORIGINAL,
        BlockSelectorKind.PER_PRECISION,
    ):
        return BlockSelector(kind=kind, block_b=None)
    block_b = _require_int(document["block_b"], policy_name, "block_selector.block_b")
    if block_b <= 0 or block_b > 65536 or block_b & (block_b - 1):
        raise _policy_error(
            policy_name,
            "block_selector.block_b must be a power of two from 1 to 65536",
        )
    return BlockSelector(kind=kind, block_b=block_b)


def validate_launch_policy_document(
    policy_name: str,
    document: object,
) -> LoadedLaunchPolicy:
    """Validate one already-decoded package policy document."""

    policy_name = _validate_policy_name(policy_name)
    if policy_name == DEFAULT_POLICY_NAME:
        raise _policy_error(policy_name, "the name is reserved for the built-in launch")
    root = _require_object(document, policy_name, "document")
    _require_exact_keys(
        root,
        required=_ROOT_REQUIRED_KEYS,
        optional=_ROOT_OPTIONAL_KEYS,
        policy_name=policy_name,
        context="document",
    )
    schema_version = _require_int(root["schema_version"], policy_name, "schema_version")
    if schema_version != 1:
        raise _policy_error(policy_name, "schema_version must be 1")
    document_name = _require_string(root["name"], policy_name, "name")
    if document_name != policy_name:
        raise _policy_error(
            policy_name,
            f"name {document_name!r} does not match the requested policy name",
        )
    operation = _require_string(root["operation"], policy_name, "operation")
    if operation != "pz_backward":
        raise _policy_error(policy_name, "operation must be 'pz_backward'")
    architecture = _parse_architecture_match(policy_name, root["architecture"])
    match = _parse_policy_match(policy_name, root["match"])
    block_selector = _parse_block_selector(policy_name, root["block_selector"])

    precision_document = _require_object(root["precisions"], policy_name, "precisions")
    _require_exact_keys(
        precision_document,
        required=_PRECISION_NAMES,
        policy_name=policy_name,
        context="precisions",
    )
    launch_metadata = MappingProxyType(
        {
            PrecisionKind(precision_name): _parse_launch_metadata(
                policy_name,
                precision_name,
                precision_document[precision_name],
                requires_block=(block_selector.kind is BlockSelectorKind.PER_PRECISION),
            )
            for precision_name in sorted(_PRECISION_NAMES)
        }
    )
    if architecture.kind is MatchKind.ANY and any(
        metadata.waves_per_eu for metadata in launch_metadata.values()
    ):
        # Only HIP accepts waves_per_eu; CUDA drops it, so a policy that can
        # apply anywhere would launch differently from what was measured.
        raise _policy_error(
            policy_name,
            "a policy for any architecture cannot set waves_per_eu",
        )
    description = root.get("description")
    if description is not None:
        description = _require_string(description, policy_name, "description")
    return LoadedLaunchPolicy(
        schema_version=schema_version,
        name=document_name,
        operation=operation,
        architecture=architecture,
        match=match,
        block_selector=block_selector,
        launch_metadata=launch_metadata,
        description=description,
    )


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last of two equal keys, which would let a stale
    # block silently override the one the validator is meant to check.
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateKeyError(key)
        document[key] = value
    return document


@lru_cache(maxsize=None)
def load_launch_policy(policy_name: str) -> LoadedLaunchPolicy:
    """Load and validate one named policy from package resources."""

    policy_name = _validate_policy_name(policy_name)
    if policy_name == DEFAULT_POLICY_NAME:
        raise LaunchPolicyNotFoundError(
            "'default' is the solver's built-in launch, not a policy document"
        )
    filename = f"{policy_name}.json"
    try:
        text = (
            resources.files(_POLICY_PACKAGE)
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as error:
        raise LaunchPolicyNotFoundError(
            f"launch policy {policy_name!r} does not exist"
        ) from error
    except OSError as error:
        raise LaunchPolicyError(
            f"failed to read launch policy {policy_name!r}: {error}"
        ) from error
    try:
        document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKeyError as error:
        raise LaunchPolicyValidationError(
            f"launch policy {policy_name!r} is invalid: duplicate key {error.key!r}"
        ) from error
    except json.JSONDecodeError as error:
        raise LaunchPolicyValidationError(
            f"launch policy {policy_name!r} contains invalid JSON: {error.msg}"
        ) from error
    return validate_launch_policy_document(policy_name, document)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def padded_rows(batch: int, block_b: int) -> int:
    return _ceil_div(batch, block_b) * block_b


def normalized_architecture(architecture: str | None) -> str:
    return (architecture or "unknown").split(":", 1)[0]


def _shape_selection(n_oi: int, batch: int, selected_block: int) -> ShapeSelection:
    baseline_block = _select_block_b(n_oi, batch)
    n_b_blocks = _ceil_div(batch, selected_block)
    return ShapeSelection(
        baseline_block=baseline_block,
        selected_block=selected_block,
        baseline_padded_rows=padded_rows(batch, baseline_block),
        selected_padded_rows=padded_rows(batch, selected_block),
        n_b_blocks=n_b_blocks,
        n_programs=n_oi * n_b_blocks,
        promoted=selected_block != baseline_block,
    )


def _require_architecture(policy: LoadedLaunchPolicy, architecture: str) -> None:
    expected = policy.architecture.value
    if policy.architecture.kind is MatchKind.EXACT and expected != architecture:
        raise LaunchPolicyMismatchError(
            f"launch policy {policy.name!r} is for architecture {expected!r}, "
            f"but this device is {architecture!r}"
        )


def _matches_call(
    policy: LoadedLaunchPolicy,
    *,
    n_oi: int,
    batch: int,
    precision: PrecisionKind | None,
    reps: int,
    preacts_trainable: bool,
    fast_measure: bool,
) -> bool:
    # Degenerate calls (empty batch, no outputs, reps=0) stay on the built-in
    # launch, which handles them.
    if n_oi <= 0 or batch <= 0 or reps <= 0:
        return False
    if precision is None or precision not in policy.launch_metadata:
        return False
    match = policy.match
    if match.kind is MatchKind.ANY:
        return True
    return (n_oi, batch, reps, preacts_trainable, fast_measure) == (
        match.n_oi,
        match.batch,
        match.reps,
        match.preacts_trainable,
        match.fast_measure,
    )


@lru_cache(maxsize=None)
def _warn_policy_fallback(policy_name: str) -> None:
    """Warn once per policy that some calls keep the default launch."""

    match = load_launch_policy(policy_name).match
    if match.kind is MatchKind.EXACT:
        tuned_for = (
            f"n_oi={match.n_oi}, batch={match.batch}, reps={match.reps}, "
            f"preacts_trainable={match.preacts_trainable}, "
            f"fast_measure={match.fast_measure}"
        )
    else:
        tuned_for = "any shape"
    warnings.warn(
        f"launch policy {policy_name!r} is tuned for {tuned_for} in fp32, bf16 "
        "or fp8; PZ backward calls outside that keep the default launch",
        RuntimeWarning,
        stacklevel=2,
    )


def _select_policy_block(
    policy: LoadedLaunchPolicy,
    precision: PrecisionKind,
    n_oi: int,
    batch: int,
) -> int:
    if policy.block_selector.kind is BlockSelectorKind.ORIGINAL:
        return _select_block_b(n_oi, batch)
    if policy.block_selector.kind is BlockSelectorKind.PER_PRECISION:
        block_b = policy.launch_metadata[precision].block_b
        if block_b is None:
            raise LaunchPolicyValidationError(
                f"launch policy {policy.name!r} has no block_b for "
                f"precision {precision.value!r}"
            )
        return block_b
    if policy.block_selector.block_b is None:
        raise LaunchPolicyValidationError(
            f"launch policy {policy.name!r} has no fixed block_b"
        )
    return policy.block_selector.block_b


def select_pz_backward_policy(
    *,
    n_oi: int,
    batch: int,
    architecture: str | None,
    precision: PrecisionKind | None,
    reps: int,
    preacts_trainable: bool,
    fast_measure: bool,
    policy_name: str | None = None,
) -> PzBackwardPolicy | None:
    """Return the launch a named policy prescribes for one PZ backward call.

    ``None`` means the built-in launch: either no policy is named (``None`` or
    ``"default"``), or the named policy was tuned for another shape, mode or
    precision than this call (the first such call per policy warns).  Pass
    ``precision=None`` for dtypes that no policy covers, such as float64.

    Raises:
        LaunchPolicyNotFoundError: no packaged document has this name.
        LaunchPolicyValidationError: the name or the document is invalid.
        LaunchPolicyMismatchError: the policy is for another architecture.
    """

    resolved_policy_name = normalize_policy_name(policy_name)
    if resolved_policy_name == DEFAULT_POLICY_NAME:
        return None
    loaded_policy = load_launch_policy(resolved_policy_name)
    arch = normalized_architecture(architecture)
    _require_architecture(loaded_policy, arch)
    if precision is None or not _matches_call(
        loaded_policy,
        n_oi=n_oi,
        batch=batch,
        precision=precision,
        reps=reps,
        preacts_trainable=preacts_trainable,
        fast_measure=fast_measure,
    ):
        _warn_policy_fallback(resolved_policy_name)
        return None
    selected_block = _select_policy_block(
        loaded_policy,
        precision,
        n_oi,
        batch,
    )
    metadata = loaded_policy.launch_metadata[precision]
    launch = LaunchConfig(
        block_b=selected_block,
        num_warps=metadata.num_warps,
        waves_per_eu=metadata.waves_per_eu,
        num_stages=metadata.num_stages,
    )
    return PzBackwardPolicy(
        architecture=arch,
        precision=precision,
        selection=_shape_selection(n_oi, batch, selected_block),
        launch=launch,
        policy_name=resolved_policy_name,
    )
