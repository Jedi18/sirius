import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.classify import Verdict, classify_gpu_error, normalize_reason


class ClassifyTests(unittest.TestCase):
    def test_plan_fallback(self):
        v, reason = classify_gpu_error(
            "Not implemented Error: GPU plan generation failed: Window not supported"
        )
        self.assertEqual(v, Verdict.PLAN_FALLBACK)
        self.assertEqual(reason, "Window not supported")

    def test_runtime_error(self):
        v, reason = classify_gpu_error(
            "Invalid Input Error: Sirius GPU execution failed: something odd"
        )
        self.assertEqual(v, Verdict.GPU_ERROR)
        self.assertEqual(reason, "something odd")

    def test_internal_and_oom(self):
        v, _ = classify_gpu_error(
            "Sirius GPU execution failed: CUDA error: an illegal memory access was encountered"
        )
        self.assertEqual(v, Verdict.GPU_INTERNAL_ERROR)
        v, _ = classify_gpu_error(
            "Sirius GPU execution failed: std::bad_alloc: out_of_memory: RMM failure"
        )
        self.assertEqual(v, Verdict.GPU_OOM)

    def test_normalize(self):
        a = normalize_reason(
            'Unsupported expression in projection (falling back to CPU): year("a0"."c1") + 3'
        )
        b = normalize_reason(
            'Unsupported expression in projection (falling back to CPU): year("a7"."c9") + 12'
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
