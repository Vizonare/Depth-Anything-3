"""CPU-only supplied-feature tests at the real backbone ownership boundary.

Run with PYTHONPATH=src python -m unittest discover -s tests -v.
No transformer inference, CUDA work, timing or peak-memory claims.
"""

import unittest
import weakref
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch import nn

from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer


def metadata(tensor):
    return (tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device)


class SuppliedFeatures(DinoVisionTransformer):
    """Supply independent raw owners without retaining them in the fixture."""

    def __init__(
        self,
        sequence=3,
        registers=0,
        width=8,
        cat_token=True,
        aux_count=2,
        batch=1,
        dtype=torch.float32,
    ):
        nn.Module.__init__(self)
        self.embed_dim = 4
        self.cat_token = cat_token
        self.num_register_tokens = registers
        self.norm = nn.LayerNorm(self.embed_dim)
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor([0.7, 1.1, 0.9, 1.3]))
            self.norm.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        self.norm.to(dtype=dtype)
        shape = (batch, sequence, 1 + registers + 5, width)
        values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
        self.supplied = [((values.reshape(shape) % 137 - 63) / 17 + i).to(dtype) for i in range(4)]
        aux_shape = (*shape[:-1], self.embed_dim)
        aux_values = torch.arange(torch.tensor(aux_shape).prod().item(), dtype=torch.float32)
        self.supplied_aux = [
            (aux_values.reshape(aux_shape) / 31 + i).to(dtype) for i in range(aux_count)
        ]
        self.raw_refs = []
        self.camera_refs = []
        self.at_norm = []
        self.received = None
        self.fail_at_norm = None
        self.norm.register_forward_pre_hook(self.observe_norm)
        self.eval()

    def _get_intermediate_layers_not_chunked(self, x, n, export_feat_layers, **kwargs):
        self.received = (x, n, export_feat_layers, kwargs)
        output = []
        for supplied in self.supplied:
            raw = supplied.clone()
            camera = raw[:, :, 0]
            self.raw_refs.append(weakref.ref(raw))
            self.camera_refs.append(weakref.ref(camera))
            output.append((camera, raw))
        return output, [out.clone() for out in self.supplied_aux]

    def observe_norm(self, module, inputs):
        self.at_norm.append(tuple(ref() is not None for ref in self.raw_refs))
        if len(self.at_norm) == self.fail_at_norm:
            raise RuntimeError("supplied norm failure")


def original_boundary(model):
    """Frozen original numerical schedule; deliberately retains all raw tuples."""
    outputs = [(raw[:, :, 0], raw) for raw in model.supplied]
    camera_tokens = [out[0] for out in outputs]
    if outputs[0][1].shape[-1] == model.embed_dim:
        outputs = [model.norm(out[1]) for out in outputs]
    elif outputs[0][1].shape[-1] == model.embed_dim * 2:
        outputs = [
            torch.cat(
                [out[1][..., : model.embed_dim], model.norm(out[1][..., model.embed_dim :])],
                dim=-1,
            )
            for out in outputs
        ]
    else:
        raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
    aux = [model.norm(out) for out in model.supplied_aux]
    outputs = [out[..., 1 + model.num_register_tokens :, :] for out in outputs]
    aux = [out[..., 1 + model.num_register_tokens :, :] for out in aux]
    return tuple(zip(outputs, camera_tokens)), aux


def traced(model, call):
    """Record only primitive call metadata; never hold an input/view alias."""
    trace = []
    cat = torch.cat
    hook = model.norm.register_forward_pre_hook(
        lambda module, inputs: trace.append(("norm", metadata(inputs[0])))
    )

    def record_cat(tensors, *args, **kwargs):
        trace.append(("cat", tuple(metadata(t) for t in tensors), args, kwargs))
        return cat(tensors, *args, **kwargs)

    try:
        with patch("torch.cat", record_cat):
            result = call()
    finally:
        hook.remove()
    return result, trace


class FeatureStorageLifetimeTests(unittest.TestCase):
    def assert_results_equal(self, expected, actual):
        self.assertIsInstance(actual[0], tuple)
        self.assertIsInstance(actual[1], list)
        self.assertEqual(len(expected[0]), len(actual[0]))
        for (ep, ec), (ap, ac) in zip(expected[0], actual[0]):
            self.assertTrue(torch.equal(ep, ap))
            self.assertTrue(torch.equal(ec, ac))
            self.assertEqual(metadata(ep), metadata(ap))
            self.assertEqual(ep.storage_offset(), ap.storage_offset())
            self.assertEqual((ec.shape, ec.dtype, ec.device), (ac.shape, ac.dtype, ac.device))
            self.assertEqual(ep.untyped_storage().nbytes(), ap.untyped_storage().nbytes())
        self.assertEqual(len(expected[1]), len(actual[1]))
        for e, a in zip(expected[1], actual[1]):
            self.assertTrue(torch.equal(e, a))
            self.assertEqual(metadata(e), metadata(a))
            self.assertEqual(e.storage_offset(), a.storage_offset())

    def test_exact_values_calls_rng_and_early_reclamation(self):
        cases = [
            (3, 0, 0, 2, torch.float32),
            (300, 0, 2, 1, torch.float32),
            (204, 2, 2, 1, torch.float32),
            (1, 0, 2, 1, torch.float64),
            (3, 2, 0, 2, torch.bfloat16),
        ]
        for sequence, registers, aux_count, batch, dtype in cases:
            with self.subTest(s=sequence, registers=registers, dtype=dtype), torch.no_grad():
                baseline = SuppliedFeatures(
                    sequence, registers, aux_count=aux_count, batch=batch, dtype=dtype
                )
                candidate = SuppliedFeatures(
                    sequence, registers, aux_count=aux_count, batch=batch, dtype=dtype
                )
                before = torch.get_rng_state().clone()
                expected, base_trace = traced(baseline, lambda: original_boundary(baseline))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                actual, trace = traced(
                    candidate,
                    lambda: candidate.get_intermediate_layers(
                        None,
                        [19, 27, 33, 39],
                        export_feat_layers=[7, 11],
                        consume_feature_storage=True,
                        cam_token="fixture",
                        ref_view_strategy="first",
                    ),
                )
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assert_results_equal(expected, actual)
                self.assertEqual(base_trace, trace)
                self.assertEqual(
                    candidate.received,
                    (
                        None,
                        [19, 27, 33, 39],
                        [7, 11],
                        {"cam_token": "fixture", "ref_view_strategy": "first"},
                    ),
                )
                for i in range(4):
                    self.assertEqual(candidate.at_norm[i], (False,) * i + (True,) * (4 - i))
                self.assertTrue(all(ref() is None for ref in candidate.raw_refs))
                self.assertTrue(all(ref() is None for ref in candidate.camera_refs))
                for patch_feature, camera in actual[0]:
                    self.assertEqual(camera.storage_offset(), 0)
                    self.assertTrue(camera.is_contiguous())
                    self.assertEqual(
                        camera.untyped_storage().nbytes(), camera.numel() * camera.element_size()
                    )
                    self.assertNotEqual(
                        camera.untyped_storage().data_ptr(),
                        patch_feature.untyped_storage().data_ptr(),
                    )

    def test_inference_mode_uses_same_values_and_releases_raw_owners(self):
        with torch.inference_mode():
            baseline = SuppliedFeatures()
            candidate = SuppliedFeatures()
            expected = original_boundary(baseline)
            actual = candidate.get_intermediate_layers(None, consume_feature_storage=True)
            self.assert_results_equal(expected, actual)
            self.assertTrue(all(ref() is None for ref in candidate.raw_refs))
            self.assertTrue(all(ref() is None for ref in candidate.camera_refs))

    def test_disabled_training_grad_and_unsupported_inertia(self):
        cases = [
            ({}, False, False, 8, True),
            ({"consume_feature_storage": False}, False, False, 8, True),
            ({"consume_feature_storage": True}, True, False, 8, True),
            ({"consume_feature_storage": True}, False, True, 8, True),
            ({"consume_feature_storage": True}, False, False, 4, False),
            ({"consume_feature_storage": True}, False, False, 8, False),
        ]
        for kwargs, training, grad, width, cat_token in cases:
            with self.subTest(
                kwargs=kwargs, training=training, grad=grad, width=width, cat_token=cat_token
            ):
                baseline = SuppliedFeatures(width=width, cat_token=cat_token)
                actual_model = SuppliedFeatures(width=width, cat_token=cat_token)
                actual_model.train(training)
                with nullcontext() if grad else torch.no_grad():
                    before = torch.get_rng_state().clone()
                    expected, base_trace = traced(baseline, lambda: original_boundary(baseline))
                    actual, trace = traced(
                        actual_model,
                        lambda: actual_model.get_intermediate_layers(None, **kwargs),
                    )
                    self.assert_results_equal(expected, actual)
                    self.assertEqual(base_trace, trace)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    self.assertEqual(actual_model.at_norm[:4], [(True,) * 4] * 4)
                    for (_, ec), (_, ac) in zip(expected[0], actual[0]):
                        self.assertEqual(ec.stride(), ac.stride())
                        self.assertEqual(ec.storage_offset(), ac.storage_offset())
                        self.assertEqual(
                            ec.untyped_storage().nbytes(), ac.untyped_storage().nbytes()
                        )
                    self.assertTrue(all(ref() is not None for ref in actual_model.raw_refs))
                    self.assertTrue(all(ref() is not None for ref in actual_model.camera_refs))
                    if grad:
                        sum(p.sum() + c.sum() for p, c in actual[0]).backward()
                        self.assertIsNotNone(actual_model.norm.weight.grad)

    def test_invalid_shape_preserves_error(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), torch.no_grad():
                model = SuppliedFeatures(width=7)
                with self.assertRaisesRegex(ValueError, "Invalid output shape"):
                    model.get_intermediate_layers(None, consume_feature_storage=enabled)
                self.assertEqual(model.at_norm, [])
                self.assertTrue(all(ref() is None for ref in model.raw_refs))
                self.assertTrue(all(ref() is None for ref in model.camera_refs))

    def test_exception_releases_owners_and_next_call_is_ordinary(self):
        with torch.no_grad():
            for enabled in (False, True):
                model = SuppliedFeatures()
                model.fail_at_norm = 2
                before = torch.get_rng_state().clone()
                with self.assertRaisesRegex(RuntimeError, "supplied norm failure"):
                    model.get_intermediate_layers(None, consume_feature_storage=enabled)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertTrue(all(ref() is None for ref in model.raw_refs))
                self.assertTrue(all(ref() is None for ref in model.camera_refs))
                model.fail_at_norm = None
                model.raw_refs.clear()
                model.camera_refs.clear()
                model.at_norm.clear()
                model.get_intermediate_layers(None)
                self.assertEqual(model.at_norm[:4], [(True,) * 4] * 4)


if __name__ == "__main__":
    unittest.main()
