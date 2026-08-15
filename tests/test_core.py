import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mg_rl.dca import net_benefit
from mg_rl.parsing import extract_grade, has_strict_structure
from mg_rl.reward_formula import composite_reward
from mg_rl.rewards_recovered import (
    criteria_score,
    diagnosis_score,
    format_score,
)


class ParsingTests(unittest.TestCase):
    def test_strict_grade(self):
        self.assertEqual(extract_grade("<answer>\n3\n</answer>"), 3)

    def test_structure(self):
        self.assertTrue(
            has_strict_structure(
                "<think>- finding</think><answer>2</answer>"
            )
        )
        self.assertFalse(
            has_strict_structure(
                "prefix <think>x</think><answer>2</answer>"
            )
        )


class RewardTests(unittest.TestCase):
    def test_recovered_diagnosis_reward(self):
        self.assertEqual(
            diagnosis_score("<answer>4</answer>", "<answer>4</answer>"),
            1.2,
        )
        self.assertEqual(
            diagnosis_score("<answer>0</answer>", "<answer>4</answer>"),
            -0.4,
        )

    def test_recovered_format_reward(self):
        self.assertEqual(
            format_score("<think>x</think><answer>2</answer>"), 1.2
        )

    def test_criteria_cap(self):
        text = (
            "microaneurysms, hemorrhage, hard exudates, venous beading, "
            "IRMA, neovascularization"
        )
        self.assertEqual(criteria_score(text), 0.5)

    def test_manuscript_composite(self):
        self.assertAlmostEqual(composite_reward(1.0, 1.0, 0.5), 0.8)


class DecisionCurveTests(unittest.TestCase):
    def test_net_benefit(self):
        import numpy as np

        truth = np.array([1, 1, 0, 0])
        decision = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(net_benefit(truth, decision, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
