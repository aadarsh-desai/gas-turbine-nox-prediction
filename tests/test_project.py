"""Checks for leakage boundaries, saved-model inference, and the working demo."""
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from turbine.config import (ARTIFACT_DIR, FEATURES, ROOT, TARGET, TEST_YEARS,
                            TRAIN_YEARS, VALIDATION_YEARS)
from turbine.data import fit_scaler, load_years
from turbine.inference import load_model, predict, validate_inputs
from turbine.metrics import regression_metrics


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.checkpoint = load_model()
        cls.examples = pd.read_csv(ARTIFACT_DIR / "demo_inputs.csv")

    def test_split_boundaries_and_train_only_scaling(self):
        self.assertLess(max(TRAIN_YEARS), min(VALIDATION_YEARS))
        self.assertLess(max(VALIDATION_YEARS), min(TEST_YEARS))
        training = load_years(TRAIN_YEARS)
        self.assertEqual(len(training), 15039)
        self.assertEqual(len(load_years(VALIDATION_YEARS)), 7152)
        self.assertEqual(len(load_years(TEST_YEARS)), 14542)
        self.assertEqual(fit_scaler(training), self.checkpoint["scaler"])
        self.assertNotIn(TARGET, FEATURES)
        self.assertNotIn("CO", FEATURES)
        # Exact repeated rows must not cross the chronological split boundaries.
        keys = [set(pd.util.hash_pandas_object(load_years(years)[FEATURES + [TARGET]],
                                               index=False).to_list())
                for years in [TRAIN_YEARS, VALIDATION_YEARS, TEST_YEARS]]
        for first, second in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(keys[first] & keys[second])

    def test_reload_predictions_and_column_order(self):
        expected = predict(self.examples, self.model, self.checkpoint)
        fresh_model, fresh_checkpoint = load_model()
        reloaded = predict(self.examples[list(reversed(FEATURES))], fresh_model, fresh_checkpoint)
        np.testing.assert_array_equal(expected, reloaded)
        self.assertEqual(expected.shape, (len(self.examples),))
        self.assertTrue(np.isfinite(expected).all())
        self.assertFalse(self.model.training)

    def test_invalid_inputs_fail_clearly(self):
        cases = [self.examples.iloc[:0], self.examples.drop(columns="AT"),
                 self.examples.assign(AT=np.nan), self.examples.assign(AT=np.inf),
                 self.examples.assign(AT="not a measurement")]
        for frame in cases:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    validate_inputs(frame)

    def test_metrics_and_frozen_report(self):
        self.assertEqual(regression_metrics([1, 2, 3], [1, 2, 3]),
                         {"mae": 0.0, "rmse": 0.0, "r2": 1.0})
        self.assertAlmostEqual(regression_metrics([1, 2, 3], [2, 2, 2])["r2"], 0.0)
        report = json.loads((ARTIFACT_DIR / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["checkpoint_sha256"],
                         hashlib.sha256((ARTIFACT_DIR / "model.pt").read_bytes()).hexdigest())
        test = load_years(TEST_YEARS)
        computed = regression_metrics(test[TARGET], predict(test, self.model, self.checkpoint))
        for key, value in computed.items():
            self.assertAlmostEqual(value, report["models"]["Neural network"][key], places=5)

    def test_demo_predicts_without_training(self):
        before = hashlib.sha256((ARTIFACT_DIR / "model.pt").read_bytes()).hexdigest()
        # Any accidental optimizer construction during app startup fails this check.
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Demo must not train")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            initial = app.metric[0].value
            app.number_input[0].set_value(app.number_input[0].value + 3.0)
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotEqual(initial, app.metric[0].value)
        after = hashlib.sha256((ARTIFACT_DIR / "model.pt").read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
