"""Unit testing for the original SoftAdapt variant."""
import unittest

import numpy as np
from keras import backend, ops

from softadapt import SoftAdapt


class TestSoftAdapt(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(class_):
        class_.rtol = 1e-5

    # First starting with positive slope test cases.
    def test_beta_positive_three_components(self):
        loss_component1 = ops.convert_to_tensor([1, 2, 3, 4, 5], backend.floatx())
        loss_component2 = ops.convert_to_tensor([150, 100, 50, 10, 0.1], backend.floatx())
        loss_component3 = ops.convert_to_tensor([1500, 1000, 500, 100, 1], backend.floatx())

        solutions = ops.convert_to_tensor([9.9343e-01, 6.5666e-03, 3.8908e-22], backend.floatx())

        softadapt_object = SoftAdapt(beta=0.1)
        alpha_0, alpha_1, alpha_2 = softadapt_object.get_component_weights(
            loss_component1, loss_component2, loss_component3, verbose=False
        )
        assert np.isclose(
            ops.convert_to_numpy(alpha_0),
            ops.convert_to_numpy(solutions[0]),
            rtol=self.rtol),\
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case."\
            "The first loss component failed."

        assert np.isclose(
            ops.convert_to_numpy(alpha_1),
            ops.convert_to_numpy(solutions[1]),
            rtol=self.rtol),\
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case."\
            "The second loss component failed."

        assert np.isclose(
            ops.convert_to_numpy(alpha_2),
            ops.convert_to_numpy(solutions[2]),
            rtol=self.rtol),\
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case."\
            "The second loss component failed."

    # TODO: Add more sophisticated unit tests


if __name__ == "__main__":
    unittest.main()
