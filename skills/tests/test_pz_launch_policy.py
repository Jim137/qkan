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

"""CPU-only tests for the PZ backward launch-policy loader and selector."""

from __future__ import annotations

import copy
import json
import unittest
import warnings
from contextlib import contextmanager
from importlib import import_module, resources
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from qkan._kernel_utils import _select_block_b
from qkan.solver.flash import launch_policy
from qkan.solver.flash.launch_policy import (
    LaunchConfig,
    LaunchPolicyMismatchError,
    LaunchPolicyNotFoundError,
    LaunchPolicyValidationError,
    MatchKind,
    PrecisionKind,
    load_launch_policy,
    select_pz_backward_policy,
    validate_launch_policy_document,
)

POLICY_N256 = "pz-gfx942-n256-batch4096-r3-fixed-preacts-fast-v1"
POLICY_N256_B512 = "pz-gfx942-n256-batch512-r3-fixed-preacts-fast-v1"
POLICY_N10000 = "pz-gfx942-n10000-batch1000-r3-fixed-preacts-fast-v1"
POLICY_DEFAULT = "default"
PACKAGED_POLICIES = {
    POLICY_N256: (256, 4096),
    POLICY_N256_B512: (256, 512),
    POLICY_N10000: (10000, 1000),
}


class FakePolicyFiles:
    """Stand-in for ``importlib.resources.files`` serving one document."""

    def __init__(self, text: str) -> None:
        self.text = text

    def joinpath(self, filename: str) -> "FakePolicyFiles":
        return self

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.text


def policy_document(name: str) -> dict[str, object]:
    text = (
        resources.files("qkan.solver.flash.policies")
        .joinpath(f"{name}.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def select(
    *,
    n_oi: int,
    batch: int,
    precision: PrecisionKind | None = PrecisionKind.FP32,
    architecture: str | None = "gfx942:sramecc+:xnack-",
    reps: int = 3,
    preacts_trainable: bool = False,
    fast_measure: bool = True,
    policy_name: str | None = None,
):
    return select_pz_backward_policy(
        n_oi=n_oi,
        batch=batch,
        architecture=architecture,
        precision=precision,
        reps=reps,
        preacts_trainable=preacts_trainable,
        fast_measure=fast_measure,
        policy_name=policy_name,
    )


class DefaultPolicyTests(unittest.TestCase):
    """``None`` and ``"default"`` mean the solver's own launch, no policy."""

    def test_none_and_default_select_no_policy(self) -> None:
        for n_oi, batch, architecture in (
            (256, 4096, None),
            (10000, 1000, "gfx942:sramecc+:xnack-"),
            (896, 500, "gfx950"),
            (1, 33, "sm90"),
            (0, 0, "sm120"),
        ):
            for precision in (*PrecisionKind, None):
                for name in (None, POLICY_DEFAULT):
                    with self.subTest(
                        n_oi=n_oi,
                        batch=batch,
                        architecture=architecture,
                        precision=None if precision is None else precision.value,
                        name=name,
                    ):
                        self.assertIsNone(
                            select(
                                n_oi=n_oi,
                                batch=batch,
                                architecture=architecture,
                                precision=precision,
                                policy_name=name,
                            )
                        )

    def test_default_has_no_policy_document(self) -> None:
        names = {
            path.name
            for path in resources.files("qkan.solver.flash.policies").iterdir()
        }
        self.assertNotIn("default.json", names)
        with self.assertRaises(LaunchPolicyNotFoundError):
            load_launch_policy(POLICY_DEFAULT)

    def test_default_reads_no_package_resources(self) -> None:
        """A build that ships only .py files must still train with flash."""

        with patch.object(
            launch_policy.resources,
            "files",
            side_effect=AssertionError("default read a resource"),
        ):
            for name in (None, POLICY_DEFAULT):
                self.assertIsNone(select(n_oi=256, batch=4096, policy_name=name))


class PackagedPolicyTests(unittest.TestCase):
    def test_n256_policy_has_all_three_exact_configs(self) -> None:
        expected = LaunchConfig(512, 2, 1, 2)
        for precision in PrecisionKind:
            with self.subTest(precision=precision.value):
                policy = select(
                    n_oi=256,
                    batch=4096,
                    precision=precision,
                    policy_name=POLICY_N256,
                )
                self.assertEqual(policy.launch, expected)
                self.assertEqual(policy.policy_name, POLICY_N256)

    def test_n10000_policy_has_dtype_specific_exact_configs(self) -> None:
        expected = {
            PrecisionKind.FP32: LaunchConfig(1024, 2, 1, 2),
            PrecisionKind.BF16: LaunchConfig(1024, 4, 1, 2),
            PrecisionKind.FP8: LaunchConfig(1024, 2, 1, 2),
        }
        for precision, config in expected.items():
            with self.subTest(precision=precision.value):
                policy = select(
                    n_oi=10000,
                    batch=1000,
                    precision=precision,
                    policy_name=POLICY_N10000,
                )
                self.assertEqual(policy.launch, config)
                self.assertEqual(policy.policy_name, POLICY_N10000)

    def test_n256_batch512_selection_matches_its_document(self) -> None:
        document = policy_document(POLICY_N256_B512)
        block_b = document["block_selector"]["block_b"]
        for precision in PrecisionKind:
            with self.subTest(precision=precision.value):
                metadata = document["precisions"][precision.value]
                policy = select(
                    n_oi=256,
                    batch=512,
                    precision=precision,
                    policy_name=POLICY_N256_B512,
                )
                self.assertEqual(
                    policy.launch,
                    LaunchConfig(
                        block_b,
                        metadata["num_warps"],
                        metadata["waves_per_eu"],
                        metadata["num_stages"],
                    ),
                )
                self.assertEqual(policy.policy_name, POLICY_N256_B512)

    def test_tuned_policies_never_pad_state_beyond_the_baseline(self) -> None:
        """Every tuned policy must fit in the original padded-state budget."""

        for policy_name, (n_oi, batch) in PACKAGED_POLICIES.items():
            architecture = load_launch_policy(policy_name).architecture.value
            for precision in PrecisionKind:
                with self.subTest(policy=policy_name, precision=precision.value):
                    selection = select(
                        n_oi=n_oi,
                        batch=batch,
                        precision=precision,
                        architecture=architecture,
                        policy_name=policy_name,
                    ).selection
                    self.assertLessEqual(
                        selection.selected_padded_rows,
                        selection.baseline_padded_rows,
                    )

    def test_shape_and_mode_mismatches_fall_back_to_the_default(self) -> None:
        """One solver_kwargs dict reaches every layer and every batch.

        A named policy therefore meets shapes it was not tuned for (other
        layers, the short last batch of an epoch); those calls keep the
        default launch instead of failing mid-training, with one warning.
        """

        mismatches = (
            {"n_oi": 255},
            {"batch": 4095},
            {"batch": 0},
            {"n_oi": 0},
            {"reps": 2},
            {"reps": 0},
            {"preacts_trainable": True},
            {"fast_measure": False},
            {"precision": None},
        )
        launch_policy._warn_policy_fallback.cache_clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for override in mismatches:
                kwargs = {
                    "n_oi": 256,
                    "batch": 4096,
                    "architecture": "gfx942",
                    "policy_name": POLICY_N256,
                }
                kwargs.update(override)
                with self.subTest(override=override):
                    self.assertIsNone(select(**kwargs))
        launch_policy._warn_policy_fallback.cache_clear()
        self.assertEqual(len(caught), 1)
        self.assertIn(POLICY_N256, str(caught[0].message))

    def test_explicit_policy_architecture_mismatch_fails_closed(self) -> None:
        for architecture in ("gfx950", "sm90", None):
            with self.subTest(architecture=architecture):
                with self.assertRaises(LaunchPolicyMismatchError):
                    select(
                        n_oi=256,
                        batch=4096,
                        architecture=architecture,
                        policy_name=POLICY_N256,
                    )

    def test_packaged_policies_only_match_their_own_architecture(self) -> None:
        """A tuning never silently applies on another vendor or generation."""

        for policy_name, (n_oi, batch) in PACKAGED_POLICIES.items():
            own = load_launch_policy(policy_name).architecture.value
            for architecture in ("gfx942", "gfx950", "sm90", "sm100", "sm120"):
                if architecture == own:
                    continue
                with self.subTest(policy=policy_name, architecture=architecture):
                    with self.assertRaises(LaunchPolicyMismatchError):
                        select(
                            n_oi=n_oi,
                            batch=batch,
                            architecture=architecture,
                            policy_name=policy_name,
                        )

    def test_unknown_and_unsafe_names_fail_closed(self) -> None:
        with self.assertRaises(LaunchPolicyNotFoundError):
            load_launch_policy("does-not-exist")
        for name in ("../escape", "/absolute", "foo.json", "a/b", "..", "UPPER"):
            with self.subTest(name=name):
                with self.assertRaises(LaunchPolicyValidationError):
                    load_launch_policy(name)

    def test_loader_caches_one_parsed_object_per_name(self) -> None:
        load_launch_policy.cache_clear()
        original_files = launch_policy.resources.files
        with patch.object(
            launch_policy.resources,
            "files",
            wraps=original_files,
        ) as files:
            first = load_launch_policy(POLICY_N256)
            second = load_launch_policy(POLICY_N256)
        self.assertIs(first, second)
        self.assertEqual(files.call_count, 1)

    def test_every_packaged_policy_loads_and_validates(self) -> None:
        package = resources.files("qkan.solver.flash.policies")
        names = {
            path.name.removesuffix(".json")
            for path in package.iterdir()
            if path.name.endswith(".json")
        }
        self.assertEqual(names, set(PACKAGED_POLICIES))
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(load_launch_policy(name).name, name)

    def test_packaged_policies_are_architecture_scoped(self) -> None:
        """Tuned launches are measured on one architecture and stay there."""

        for name in PACKAGED_POLICIES:
            with self.subTest(name=name):
                self.assertIs(
                    load_launch_policy(name).architecture.kind, MatchKind.EXACT
                )

    def test_duplicate_keys_are_rejected(self) -> None:
        """json.loads keeps the last duplicate, which would defeat strictness."""

        text = (
            resources.files("qkan.solver.flash.policies")
            .joinpath(f"{POLICY_N256}.json")
            .read_text(encoding="utf-8")
        )
        duplicated = (
            text.replace('"batch": 4096', '"batch": 4096, "batch": 512'),
            text.replace('"match": {', '"match": {"kind": "any", ', 1),
            text.replace('"num_warps": 2', '"num_warps": 2, "num_warps": 8', 1),
        )
        for index, document in enumerate(duplicated):
            self.assertNotEqual(document, text)
            with self.subTest(index=index):
                load_launch_policy.cache_clear()
                with patch.object(
                    launch_policy.resources,
                    "files",
                    return_value=FakePolicyFiles(document),
                ):
                    with self.assertRaisesRegex(
                        LaunchPolicyValidationError, "duplicate key"
                    ):
                        load_launch_policy(POLICY_N256)
        load_launch_policy.cache_clear()

    def test_architecture_scoped_policy_is_rejected_on_a_foreign_vendor(self) -> None:
        with self.assertRaises(LaunchPolicyMismatchError):
            select_pz_backward_policy(
                n_oi=256,
                batch=512,
                architecture="sm90",
                precision=PrecisionKind.FP32,
                reps=3,
                preacts_trainable=False,
                fast_measure=True,
                policy_name=POLICY_N256_B512,
            )

    def test_packaged_documents_have_no_runtime_provenance(self) -> None:
        package = resources.files("qkan.solver.flash.policies")
        for name in PACKAGED_POLICIES:
            text = package.joinpath(f"{name}.json").read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertNotIn("iteration-", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn('"provenance"', text)
                self.assertNotIn('"source"', text)


class SchemaValidationTests(unittest.TestCase):
    def test_per_precision_blocks_are_strict_and_selectable(self) -> None:
        document = policy_document(POLICY_N256)
        name = "synthetic-per-precision"
        document["name"] = name
        document["block_selector"] = {"kind": "per_precision"}
        expected = {
            "fp32": 32,
            "bf16": 64,
            "fp8": 128,
        }
        for precision_name, block_b in expected.items():
            document["precisions"][precision_name]["block_b"] = block_b
        loaded = validate_launch_policy_document(name, document)
        self.assertEqual(
            {
                precision.value: metadata.block_b
                for precision, metadata in loaded.launch_metadata.items()
            },
            expected,
        )
        with patch.object(
            launch_policy,
            "load_launch_policy",
            return_value=loaded,
        ):
            for precision in PrecisionKind:
                with self.subTest(precision=precision.value):
                    policy = select(
                        n_oi=256,
                        batch=4096,
                        precision=precision,
                        policy_name=name,
                    )
                    self.assertEqual(
                        policy.selection.selected_block,
                        expected[precision.value],
                    )

    def test_schema_rejects_unknown_missing_and_invalid_fields(self) -> None:
        base = policy_document(POLICY_N256)
        mutations = []

        unknown = copy.deepcopy(base)
        unknown["unexpected"] = True
        mutations.append(unknown)

        missing = copy.deepcopy(base)
        del missing["match"]
        mutations.append(missing)

        wrong_version = copy.deepcopy(base)
        wrong_version["schema_version"] = 2
        mutations.append(wrong_version)

        wrong_name = copy.deepcopy(base)
        wrong_name["name"] = "different-name"
        mutations.append(wrong_name)

        wrong_operation = copy.deepcopy(base)
        wrong_operation["operation"] = "other"
        mutations.append(wrong_operation)

        bad_architecture = copy.deepcopy(base)
        bad_architecture["architecture"]["value"] = "../gfx942"
        mutations.append(bad_architecture)

        wrong_type = copy.deepcopy(base)
        wrong_type["match"]["n_oi"] = True
        mutations.append(wrong_type)

        missing_precision = copy.deepcopy(base)
        del missing_precision["precisions"]["fp8"]
        mutations.append(missing_precision)

        unknown_precision = copy.deepcopy(base)
        unknown_precision["precisions"]["fp16"] = copy.deepcopy(
            unknown_precision["precisions"]["fp32"]
        )
        mutations.append(unknown_precision)

        bad_block = copy.deepcopy(base)
        bad_block["block_selector"]["block_b"] = 300
        mutations.append(bad_block)

        block_in_metadata = copy.deepcopy(base)
        block_in_metadata["precisions"]["fp32"]["block_b"] = 512
        mutations.append(block_in_metadata)

        unknown_selector = copy.deepcopy(base)
        unknown_selector["block_selector"]["kind"] = "adaptive"
        mutations.append(unknown_selector)

        original_with_block = copy.deepcopy(base)
        original_with_block["block_selector"] = {"kind": "original", "block_b": 128}
        mutations.append(original_with_block)

        any_match_with_shape = copy.deepcopy(base)
        any_match_with_shape["match"]["kind"] = "any"
        mutations.append(any_match_with_shape)

        # A policy that can apply on any vendor must not need an AMD-only
        # option: CUDA drops waves_per_eu, so the launch would differ.
        any_architecture_with_waves = copy.deepcopy(base)
        any_architecture_with_waves["architecture"] = {"kind": "any"}
        mutations.append(any_architecture_with_waves)

        reserved_name = copy.deepcopy(base)
        reserved_name["name"] = POLICY_DEFAULT
        mutations.append((POLICY_DEFAULT, reserved_name))

        bad_warps = copy.deepcopy(base)
        bad_warps["precisions"]["fp32"]["num_warps"] = 3
        mutations.append(bad_warps)

        bad_stages = copy.deepcopy(base)
        bad_stages["precisions"]["fp32"]["num_stages"] = 0
        mutations.append(bad_stages)

        for index, mutation in enumerate(mutations):
            name, document = (
                mutation if isinstance(mutation, tuple) else (POLICY_N256, mutation)
            )
            with self.subTest(index=index):
                with self.assertRaises(LaunchPolicyValidationError):
                    validate_launch_policy_document(name, document)

    def test_architecture_schema_is_not_vendor_hardcoded(self) -> None:
        document = policy_document(POLICY_N256)
        document["name"] = "synthetic-sm90-policy"
        document["architecture"] = {"kind": "exact", "value": "sm90"}
        loaded = validate_launch_policy_document("synthetic-sm90-policy", document)
        self.assertEqual(loaded.architecture.value, "sm90")


class ConsumerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.fused_ops = import_module("qkan.solver.flash.fused_ops")
        except ImportError as error:  # pragma: no cover - triton is optional
            self.skipTest(f"triton is unavailable: {error}")

    def _architecture(self, properties, hip_version):
        self.fused_ops._device_architecture.cache_clear()
        try:
            with patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ):
                with patch.object(torch.version, "hip", hip_version):
                    return self.fused_ops._device_architecture("cuda", 0)
        finally:
            self.fused_ops._device_architecture.cache_clear()

    def test_cuda_architecture_is_the_sm_name_not_the_marketing_name(self) -> None:
        """CUDA builds of torch fill gcnArchName with the product name."""

        properties = SimpleNamespace(
            gcnArchName="NVIDIA GeForce RTX 5090", major=12, minor=0
        )
        self.assertEqual(self._architecture(properties, None), "sm120")
        properties = SimpleNamespace(major=9, minor=0)
        self.assertEqual(self._architecture(properties, None), "sm90")

    def test_rocm_architecture_is_the_gfx_name(self) -> None:
        properties = SimpleNamespace(
            gcnArchName="gfx942:sramecc+:xnack-", major=9, minor=4
        )
        self.assertEqual(self._architecture(properties, "6.2.0"), "gfx942")

    def _clear_launch_caches(self) -> None:
        self.fused_ops._supports_waves_per_eu.cache_clear()
        self.fused_ops._warn_waves_per_eu_dropped.cache_clear()
        self.fused_ops._triton_version.cache_clear()

    @contextmanager
    def _active_backend(self, backend: str, hip_version: str | None):
        """Run the body against Triton's real ``backend`` options object."""

        gpu_target = import_module("triton.backends.compiler").GPUTarget
        target = (
            gpu_target("hip", "gfx942", 64)
            if backend == "hip"
            else gpu_target("cuda", 90, 32)
        )
        try:
            self.fused_ops.make_backend(target)
        except Exception as error:  # pragma: no cover - backend not installed
            self.skipTest(f"triton {backend} backend unavailable: {error}")
        driver = SimpleNamespace(
            active=SimpleNamespace(get_current_target=lambda: target)
        )
        self._clear_launch_caches()
        try:
            with patch.object(self.fused_ops, "driver", driver):
                with patch.object(torch.version, "hip", hip_version):
                    yield
        finally:
            self._clear_launch_caches()

    def test_waves_per_eu_is_forwarded_where_the_backend_accepts_it(self) -> None:
        launch = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        with self._active_backend("hip", "6.2.0"):
            self.assertEqual(
                self.fused_ops._launch_options(launch),
                {"num_warps": 2, "num_stages": 2, "waves_per_eu": 3},
            )

    def test_waves_per_eu_is_dropped_where_the_backend_rejects_it(self) -> None:
        """Triton's CUDA options object has no ``waves_per_eu`` field."""

        launch = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        with self._active_backend("cuda", None):
            with self.assertWarns(RuntimeWarning):
                options = self.fused_ops._launch_options(launch)
        self.assertEqual(options, {"num_warps": 2, "num_stages": 2})

    def test_zero_waves_per_eu_is_forwarded_on_hip(self) -> None:
        """0 is a legal policy value meaning "no occupancy limit" on HIP."""

        launch = LaunchConfig(block_b=128, num_warps=4, waves_per_eu=0, num_stages=2)
        with self._active_backend("hip", "6.2.0"):
            self.assertEqual(
                self.fused_ops._launch_options(launch),
                {"num_warps": 4, "num_stages": 2, "waves_per_eu": 0},
            )

    def test_zero_waves_per_eu_is_dropped_without_warning(self) -> None:
        """Omitting 0 matches the HIP default, so it is not a divergence."""

        launch = LaunchConfig(block_b=128, num_warps=4, waves_per_eu=0, num_stages=2)
        with self._active_backend("cuda", None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                options = self.fused_ops._launch_options(launch)
        self.assertEqual(options, {"num_warps": 4, "num_stages": 2})
        self.assertEqual(caught, [])

    def test_dropped_waves_per_eu_warns_once_per_value(self) -> None:
        launch = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        with self._active_backend("cuda", None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(8):
                    self.fused_ops._launch_options(launch)
        self.assertEqual(len(caught), 1)
        self.assertIn("waves_per_eu=3", str(caught[0].message))

    def test_backend_introspection_outranks_the_torch_build_vendor(self) -> None:
        """``torch.version.hip`` describes torch, not the Triton backend."""

        launch = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        with self._active_backend("cuda", "6.2.0"):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(
                    self.fused_ops._launch_options(launch),
                    {"num_warps": 2, "num_stages": 2},
                )
        with self._active_backend("hip", None):
            self.assertEqual(
                self.fused_ops._launch_options(launch),
                {"num_warps": 2, "num_stages": 2, "waves_per_eu": 3},
            )

    def test_waves_per_eu_follows_the_triton_36_contract(self) -> None:
        """Triton 3.6 made waves_per_eu pin occupancy (min = max).

        Before 3.6 the same number was only a minimum, so a tuned non-zero
        value cannot be reproduced there and the policy is refused.
        """

        tuned = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        neutral = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=0, num_stages=2)
        with self._active_backend("hip", "6.2.0"):
            with patch.object(self.fused_ops, "_triton_version", return_value=(3, 5)):
                with self.assertRaisesRegex(LaunchPolicyMismatchError, "3.6"):
                    self.fused_ops._launch_options(tuned)
                self.assertEqual(
                    self.fused_ops._launch_options(neutral),
                    {"num_warps": 2, "num_stages": 2},
                )
            with patch.object(self.fused_ops, "_triton_version", return_value=(3, 7)):
                self.assertEqual(
                    self.fused_ops._launch_options(tuned),
                    {"num_warps": 2, "num_stages": 2, "waves_per_eu": 3},
                )

    def test_triton_version_parses_release_and_local_builds(self) -> None:
        for version, expected in (
            ("3.6.0", (3, 6)),
            ("3.5.1+rocm6.4.1.git2a2", (3, 5)),
            ("3.2.0+git4b3bb1f8", (3, 2)),
            ("nightly", (0, 0)),
        ):
            with self.subTest(version=version):
                self.fused_ops._triton_version.cache_clear()
                with patch.object(self.fused_ops.triton, "__version__", version):
                    self.assertEqual(self.fused_ops._triton_version(), expected)
        self.fused_ops._triton_version.cache_clear()

    def test_unintrospectable_driver_falls_back_to_the_torch_build(self) -> None:
        fused_ops = self.fused_ops
        driver = SimpleNamespace(
            active=SimpleNamespace(
                get_current_target=Mock(side_effect=RuntimeError("no driver"))
            )
        )
        launch = LaunchConfig(block_b=512, num_warps=2, waves_per_eu=3, num_stages=2)
        cases = (
            ("6.2.0", {"num_warps": 2, "num_stages": 2, "waves_per_eu": 3}),
            (None, {"num_warps": 2, "num_stages": 2}),
        )
        for hip_version, expected in cases:
            with self.subTest(hip=hip_version):
                self._clear_launch_caches()
                with patch.object(fused_ops, "driver", driver):
                    with patch.object(torch.version, "hip", hip_version):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            options = fused_ops._launch_options(launch)
                self.assertEqual(options, expected)
        self._clear_launch_caches()

    def test_public_autograd_path_carries_the_resolved_launch(self) -> None:
        """The policy is resolved in forward and handed to backward as data."""

        flash_module = import_module("qkan.solver.flash.flash")
        launch = self.fused_ops.PzLaunch(512, (("num_stages", 2), ("num_warps", 2)))
        x = torch.zeros(2, 1, requires_grad=True)
        theta = torch.zeros(1, 1, 4, 2, requires_grad=True)
        preacts_weight = torch.ones(1, 1, 3)
        preacts_bias = torch.zeros(1, 1, 3)
        sentinel = torch.empty(0)

        def solve(**kwargs):
            return flash_module.flash_exact_solver(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                3,
                ansatz="pz",
                out_dim=1,
                dtype=torch.float32,
                **kwargs,
            )

        with patch.object(
            flash_module, "resolve_pz_launch", return_value=launch
        ) as resolve:
            with patch.object(
                flash_module._FlashFunction, "apply", return_value=sentinel
            ) as apply:
                self.assertIs(solve(pz_launch_policy=POLICY_N256), sentinel)
                self.assertIs(apply.call_args.args[-1], launch)
                self.assertEqual(resolve.call_args.args[0], POLICY_N256)
                self.assertEqual(
                    resolve.call_args.kwargs,
                    {
                        "n_oi": 1,
                        "batch": 2,
                        "reps": 3,
                        "c_dtype": torch.float32,
                        "preacts_trainable": False,
                        "fast_measure": True,
                    },
                )

                resolve.reset_mock()
                for kwargs in ({}, {"pz_launch_policy": POLICY_DEFAULT}):
                    solve(**kwargs)
                    self.assertIsNone(apply.call_args.args[-1])
                resolve.assert_not_called()

                # No backward under no_grad, so nothing is resolved.
                with torch.no_grad():
                    with patch.object(
                        flash_module, "triton_pz_forward", return_value=sentinel
                    ):
                        solve(pz_launch_policy=POLICY_N256)
                resolve.assert_not_called()

        ctx = SimpleNamespace(
            saved_tensors=(x, theta, preacts_weight, preacts_bias),
            ansatz="pz",
            preacts_trainable=False,
            fast_measure=True,
            c_dtype=torch.float32,
            pz_launch=launch,
        )
        gradients = (torch.zeros_like(x), torch.zeros_like(theta), None, None)
        with patch.object(
            flash_module, "triton_pz_backward", return_value=gradients
        ) as backward:
            returned = flash_module._FlashFunction.backward(ctx, torch.zeros(2, 1, 1))
        self.assertIs(backward.call_args.kwargs["launch"], launch)
        self.assertEqual(len(returned), 11)

    def test_default_policy_name_is_accepted_for_every_ansatz(self) -> None:
        """``"default"`` means no policy, which every ansatz supports."""

        flash_module = import_module("qkan.solver.flash.flash")
        sentinel = torch.empty(0)
        for ansatz, forward in (
            ("rpz", "triton_rpz_forward"),
            ("real", "triton_real_forward"),
        ):
            with self.subTest(ansatz=ansatz):
                with patch.object(flash_module, forward, return_value=sentinel):
                    with torch.no_grad():
                        result = flash_module.flash_exact_solver(
                            torch.zeros(2, 1),
                            torch.zeros(1, 1, 4, 2),
                            torch.ones(1, 1, 3),
                            torch.zeros(1, 1, 3),
                            3,
                            ansatz=ansatz,
                            out_dim=1,
                            dtype=torch.float32,
                            pz_launch_policy=POLICY_DEFAULT,
                        )
                self.assertIs(result, sentinel)

    def _record_backward_launch(self, launch):
        """Run triton_pz_backward on CPU with the kernel replaced by a spy."""

        calls = []

        class Spy:
            def __getitem__(self, grid):
                def record(*args, **kwargs):
                    calls.append((grid, kwargs))

                return record

        batch, in_dim, out_dim, reps = 100, 16, 16, 3
        with patch.object(self.fused_ops, "_pz_encoding_backward_kernel", Spy()):
            self.fused_ops.triton_pz_backward(
                torch.zeros(batch, in_dim),
                torch.zeros(out_dim, in_dim, reps + 1, 2),
                torch.ones(out_dim, in_dim, reps),
                torch.zeros(out_dim, in_dim, reps),
                torch.zeros(batch, out_dim, in_dim),
                False,
                True,
                c_dtype=torch.float32,
                launch=launch,
            )
        (call,) = calls
        return call

    def test_default_launch_leaves_launch_options_to_the_backend(self) -> None:
        grid, kwargs = self._record_backward_launch(None)
        block_b = _select_block_b(256, 100)
        self.assertEqual(kwargs["BLOCK_B"], block_b)
        self.assertEqual(grid, (256, -(-100 // block_b)))
        for option in ("num_warps", "num_stages", "waves_per_eu"):
            self.assertNotIn(option, kwargs)

    def test_policy_launch_applies_its_block_and_options(self) -> None:
        launch = self.fused_ops.PzLaunch(32, (("num_stages", 1), ("num_warps", 2)))
        grid, kwargs = self._record_backward_launch(launch)
        self.assertEqual(grid, (256, 4))
        self.assertEqual(kwargs["BLOCK_B"], 32)
        self.assertEqual(kwargs["num_warps"], 2)
        self.assertEqual(kwargs["num_stages"], 1)

    def test_named_policy_is_resolved_once_per_call_shape(self) -> None:
        document = policy_document(POLICY_N256)
        document["name"] = "synthetic-cpu-policy"
        document["architecture"] = {"kind": "exact", "value": "unknown"}
        for metadata in document["precisions"].values():
            metadata["waves_per_eu"] = 0
        loaded = validate_launch_policy_document("synthetic-cpu-policy", document)
        resolve = self.fused_ops.resolve_pz_launch
        kwargs = dict(
            n_oi=256,
            batch=4096,
            reps=3,
            c_dtype=torch.float32,
            preacts_trainable=False,
            fast_measure=True,
        )
        self.fused_ops._resolve_pz_launch.cache_clear()
        try:
            with patch.object(
                launch_policy, "load_launch_policy", return_value=loaded
            ) as load:
                first = resolve("synthetic-cpu-policy", torch.device("cpu"), **kwargs)
                second = resolve("synthetic-cpu-policy", torch.device("cpu"), **kwargs)
        finally:
            self.fused_ops._resolve_pz_launch.cache_clear()
        self.assertIs(first, second)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(first.block_b, 512)
        self.assertEqual(dict(first.options)["num_warps"], 2)

    def test_policy_name_is_rejected_for_non_pz_ansatz(self) -> None:
        flash_module = import_module("qkan.solver.flash.flash")
        with self.assertRaises(ValueError):
            flash_module.flash_exact_solver(
                torch.zeros(2, 1),
                torch.zeros(1, 1, 4, 2),
                torch.ones(1, 1, 3),
                torch.zeros(1, 1, 3),
                3,
                ansatz="rpz",
                out_dim=1,
                dtype=torch.float32,
                pz_launch_policy=POLICY_N256,
            )


if __name__ == "__main__":
    unittest.main()
