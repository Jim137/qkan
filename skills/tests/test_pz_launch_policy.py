# Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
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
    BlockSelectorKind,
    LaunchConfig,
    LaunchPolicyMismatchError,
    LaunchPolicyNotFoundError,
    LaunchPolicyValidationError,
    MatchKind,
    PrecisionKind,
    load_launch_policy,
    precision_plan,
    select_pz_backward_policy,
    validate_launch_policy_document,
)

POLICY_N256 = "pz-gfx942-n256-batch4096-r3-fixed-preacts-fast-v1"
POLICY_N256_B512 = "pz-gfx942-n256-batch512-r3-fixed-preacts-fast-v1"
POLICY_N10000 = "pz-gfx942-n10000-batch1000-r3-fixed-preacts-fast-v1"
POLICY_DEFAULT = "default"


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
    precision: PrecisionKind = PrecisionKind.FP32,
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


class PrecisionAndDefaultTests(unittest.TestCase):
    def test_fp8_plan_preserves_current_contract(self) -> None:
        fp8 = precision_plan(PrecisionKind.FP8)
        self.assertEqual(
            (
                fp8.kind,
                fp8.io_dtype_name,
                fp8.state_dtype_name,
                fp8.state_itemsize,
                fp8.fp8_prescale,
                fp8.launch_precision_key,
            ),
            (
                PrecisionKind.FP8,
                "bfloat16",
                "float8_e4m3fn",
                1,
                224.0,
                "fp8",
            ),
        )

    def test_none_and_explicit_default_are_identical(self) -> None:
        for n_oi, batch, architecture in (
            (256, 4096, None),
            (10000, 1000, "gfx942:sramecc+:xnack-"),
            (896, 500, "gfx950"),
            (1, 33, "sm90"),
        ):
            for precision in PrecisionKind:
                with self.subTest(
                    n_oi=n_oi,
                    batch=batch,
                    architecture=architecture,
                    precision=precision,
                ):
                    omitted = select(
                        n_oi=n_oi,
                        batch=batch,
                        architecture=architecture,
                        precision=precision,
                        policy_name=None,
                    )
                    explicit = select(
                        n_oi=n_oi,
                        batch=batch,
                        architecture=architecture,
                        precision=precision,
                        policy_name=POLICY_DEFAULT,
                    )
                    expected_block = _select_block_b(n_oi, batch)
                    self.assertEqual(omitted, explicit)
                    self.assertEqual(omitted.policy_name, POLICY_DEFAULT)
                    self.assertEqual(
                        omitted.launch,
                        LaunchConfig(expected_block, 4, 0, 2),
                    )

    def test_default_never_pads_state_beyond_the_original_selector(self) -> None:
        """The default policy must not grow the padded ``states`` footprint."""

        for n_oi, batch in (
            (256, 256),
            (256, 512),
            (256, 1000),
            (256, 4096),
            (10000, 1000),
            (65536, 256),
        ):
            with self.subTest(n_oi=n_oi, batch=batch):
                selection = select(n_oi=n_oi, batch=batch).selection
                self.assertFalse(selection.promoted)
                self.assertEqual(
                    selection.selected_padded_rows,
                    selection.baseline_padded_rows,
                )

    def test_default_keeps_the_original_program_count_for_256_pair_grids(self) -> None:
        """A 16x16 PZ layer keeps its baseline batch blocks under the default."""

        selection = select(n_oi=256, batch=1000).selection
        baseline_block = _select_block_b(256, 1000)
        self.assertEqual(selection.selected_block, baseline_block)
        self.assertEqual(
            selection.n_programs,
            256 * -(-1000 // baseline_block),
        )

    def test_default_is_loaded_and_cached_from_package_resources(self) -> None:
        load_launch_policy.cache_clear()
        original_files = launch_policy.resources.files
        with patch.object(
            launch_policy.resources,
            "files",
            wraps=original_files,
        ) as files:
            omitted = select(n_oi=256, batch=4096, policy_name=None)
            explicit = select(
                n_oi=256,
                batch=4096,
                policy_name=POLICY_DEFAULT,
            )
        loaded = load_launch_policy(POLICY_DEFAULT)
        self.assertEqual(files.call_count, 1)
        self.assertEqual(omitted, explicit)
        self.assertEqual(loaded.architecture.kind, MatchKind.ANY)
        self.assertEqual(loaded.match.kind, MatchKind.ANY)
        self.assertEqual(
            loaded.block_selector.kind,
            BlockSelectorKind.ORIGINAL,
        )


class PackagedPolicyTests(unittest.TestCase):
    def test_n256_policy_has_all_three_exact_configs(self) -> None:
        expected = LaunchConfig(512, 2, 1, 2)
        for precision in PrecisionKind:
            with self.subTest(precision=precision):
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
            with self.subTest(precision=precision):
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
            with self.subTest(precision=precision):
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

        shapes = {
            POLICY_N256_B512: (256, 512),
            POLICY_N256: (256, 4096),
            POLICY_N10000: (10000, 1000),
        }
        for policy_name, (n_oi, batch) in shapes.items():
            for precision in PrecisionKind:
                with self.subTest(policy=policy_name, precision=precision):
                    selection = select(
                        n_oi=n_oi,
                        batch=batch,
                        precision=precision,
                        policy_name=policy_name,
                    ).selection
                    self.assertLessEqual(
                        selection.selected_padded_rows,
                        selection.baseline_padded_rows,
                    )

    def test_explicit_policy_runtime_mismatch_fails_closed(self) -> None:
        mismatches = (
            {"architecture": "gfx950"},
            {"n_oi": 255},
            {"batch": 4095},
            {"reps": 2},
            {"preacts_trainable": True},
            {"fast_measure": False},
        )
        for override in mismatches:
            kwargs = {
                "n_oi": 256,
                "batch": 4096,
                "architecture": "gfx942",
                "reps": 3,
                "preacts_trainable": False,
                "fast_measure": True,
                "policy_name": POLICY_N256,
            }
            kwargs.update(override)
            with self.subTest(override=override):
                with self.assertRaises(LaunchPolicyMismatchError):
                    select(**kwargs)

    def test_gfx942_policies_do_not_match_a_cuda_runtime(self) -> None:
        """AMD-tuned policies must never silently apply on NVIDIA."""

        shapes = {
            POLICY_N256_B512: (256, 512),
            POLICY_N256: (256, 4096),
            POLICY_N10000: (10000, 1000),
        }
        for policy_name, (n_oi, batch) in shapes.items():
            with self.subTest(policy=policy_name):
                with self.assertRaises(LaunchPolicyMismatchError):
                    select(
                        n_oi=n_oi,
                        batch=batch,
                        architecture="sm90",
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
        self.assertEqual(
            names,
            {POLICY_DEFAULT, POLICY_N256, POLICY_N256_B512, POLICY_N10000},
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(load_launch_policy(name).name, name)

    def test_vendor_agnostic_policies_do_not_rely_on_amd_only_options(self) -> None:
        """A policy selectable on any vendor must survive dropping HIP options.

        Architecture-scoped policies cannot reach a foreign backend because
        selection rejects them, so only ``kind: any`` policies must hold to
        ``waves_per_eu == 0``, which is exactly what omitting it means.
        """

        package = resources.files("qkan.solver.flash.policies")
        agnostic = [
            policy
            for path in package.iterdir()
            if path.name.endswith(".json")
            for policy in [load_launch_policy(path.name.removesuffix(".json"))]
            if policy.architecture.kind is MatchKind.ANY
        ]
        self.assertEqual([policy.name for policy in agnostic], [POLICY_DEFAULT])
        for policy in agnostic:
            for precision, metadata in policy.launch_metadata.items():
                with self.subTest(policy=policy.name, precision=precision):
                    self.assertEqual(metadata.waves_per_eu, 0)

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
        names = {POLICY_DEFAULT, POLICY_N256, POLICY_N256_B512, POLICY_N10000}
        for name in names:
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
                with self.subTest(precision=precision):
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

        original_with_block = policy_document(POLICY_DEFAULT)
        original_with_block["block_selector"]["block_b"] = 128
        mutations.append(original_with_block)

        any_match_with_shape = policy_document(POLICY_DEFAULT)
        any_match_with_shape["match"]["n_oi"] = 256
        mutations.append(any_match_with_shape)

        bad_warps = copy.deepcopy(base)
        bad_warps["precisions"]["fp32"]["num_warps"] = 3
        mutations.append(bad_warps)

        bad_stages = copy.deepcopy(base)
        bad_stages["precisions"]["fp32"]["num_stages"] = 0
        mutations.append(bad_stages)

        for index, document in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(LaunchPolicyValidationError):
                    validate_launch_policy_document(POLICY_N256, document)

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

    def test_device_architecture_detection_supports_cuda_sm_names(self) -> None:
        self.fused_ops._device_properties_snapshot.cache_clear()
        properties = SimpleNamespace(
            major=9,
            minor=0,
            multi_processor_count=132,
        )
        with patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=properties,
        ):
            self.assertEqual(
                self.fused_ops._device_properties_snapshot("cuda", 0),
                ("sm90", 132),
            )
        self.fused_ops._device_properties_snapshot.cache_clear()

    def _clear_launch_caches(self) -> None:
        self.fused_ops._supports_waves_per_eu.cache_clear()
        self.fused_ops._warn_waves_per_eu_dropped.cache_clear()

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

    def test_public_autograd_path_carries_policy_name_to_backward(self) -> None:
        flash_module = import_module("qkan.solver.flash.flash")
        x = torch.zeros(2, 1, requires_grad=True)
        theta = torch.zeros(1, 1, 4, 2, requires_grad=True)
        preacts_weight = torch.ones(1, 1, 3)
        preacts_bias = torch.zeros(1, 1, 3)
        sentinel = torch.empty(0)
        with patch.object(
            flash_module._FlashFunction,
            "apply",
            return_value=sentinel,
        ) as apply:
            result = flash_module.flash_exact_solver(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                3,
                ansatz="pz",
                out_dim=1,
                dtype=torch.float32,
                pz_launch_policy=POLICY_N256,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(apply.call_args.args[-1], POLICY_N256)

        with patch.object(
            flash_module._FlashFunction,
            "apply",
            return_value=sentinel,
        ) as default_apply:
            flash_module.flash_exact_solver(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                3,
                ansatz="pz",
                out_dim=1,
                dtype=torch.float32,
            )
        self.assertIsNone(default_apply.call_args.args[-1])

        ctx = SimpleNamespace(
            saved_tensors=(x, theta, preacts_weight, preacts_bias),
            ansatz="pz",
            preacts_trainable=False,
            fast_measure=True,
            c_dtype=torch.float32,
            pz_launch_policy=POLICY_N256,
        )
        gradients = (
            torch.zeros_like(x),
            torch.zeros_like(theta),
            None,
            None,
        )
        with patch.object(
            flash_module,
            "triton_pz_backward",
            return_value=gradients,
        ) as backward:
            returned = flash_module._FlashFunction.backward(ctx, torch.zeros(2, 1, 1))
        self.assertEqual(backward.call_args.kwargs["policy_name"], POLICY_N256)
        self.assertEqual(len(returned), 11)

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
