"""Regression for Windows CRLF checkouts of LightGBM text models."""
import unittest
from service.common import ROOT
import lightgbm as lgb
import numpy as np


class SavedModelTests(unittest.TestCase):
    def test_normalized_saved_model_loads_and_scores(self):
        text=(ROOT/'models/lightgbm_baseline_stratified.txt').read_text(encoding='utf-8')
        self.assertNotIn('\r',text)
        model=lgb.Booster(model_str=text)
        score=model.predict(np.zeros((1,model.num_feature())),num_threads=1)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(0<=score[0]<=1)
