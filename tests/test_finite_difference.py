"""Unit testing for calculating finite difference approximations."""

import unittest

import numpy as np

from softadapt.utilities._finite_difference import _get_finite_difference


class TestFiniteDifference(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(class_):
        class_.rtol = 1e-5

    # First starting with positive slope test cases.
    def test_first_order_positive_slope(self):
        order = 2
        loss_points = [0, 1, 2, 3, 4, 5]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(1, approximation, rtol=self.rtol), (
            "Incorrect first order approximation for simple positive slope test case."
        )

    def test_second_order_positive_slope(self):
        order = 2
        loss_points = [0, 1, 2, 3, 4, 5]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(1, approximation, rtol=self.rtol), (
            "Incorrect second order approximation for simple negative slope test case."
        )

    def test_third_order_positive_slope(self):
        order = 3
        loss_points = [0, 2, 4, 6, 8, 10]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(2, approximation, rtol=self.rtol), (
            "Incorrect third order approximation for simple positive slope test case."
        )

    def test_fourth_order_positive_slope(self):
        order = 4
        loss_points = [0, 2, 4, 6, 8, 10]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(2, approximation, rtol=self.rtol), (
            "Incorrect fourth order approximation for simple positive slope test case."
        )

    def test_fifth_order_positive_slope(self):
        order = 5
        loss_points = [-5, -4, -3, -2, -1, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(1, approximation, rtol=self.rtol), (
            "Incorrect fifth order approximation for simple positive slope test case."
        )

    def test_tenth_order_positive_slope(self):
        order = 10
        loss_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(1, approximation, rtol=self.rtol), (
            "Incorrect 10th order approximation for simple positive slope test case."
        )

    # From here on we have negative slope test cases.
    def test_first_order_negative_slope(self):
        order = 1
        loss_points = [15, 12, 9, 6, 3, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-3, approximation, rtol=self.rtol), (
            "Incorrect first order approximation for simple negative slope test case."
        )

    def test_second_order_negative_slope(self):
        order = 2
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-1, approximation, rtol=self.rtol), (
            "Incorrect second order approximation for simple negative slope test case."
        )

    def test_third_order_negative_slope(self):
        order = 3
        loss_points = [20, 16, 12, 8, 4, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-4, approximation, rtol=self.rtol), (
            "Incorrect third order approximation for simple negative slope test case."
        )

    def test_fourth_order_negative_slope(self):
        order = 4
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-1, approximation, rtol=self.rtol), (
            "Incorrect fourth order approximation for simple negative slope test case."
        )

    def test_fifth_order_negative_slope(self):
        order = 5
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-1, approximation, rtol=self.rtol), (
            "Incorrect fifth order approximation for simple negative slope test case."
        )

    def test_tenth_order_negative_slope(self):
        order = 10
        loss_points = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        approximation = _get_finite_difference(loss_points, order)
        assert np.isclose(-1, approximation, rtol=self.rtol), (
            "Incorrect 10th order approximation for simple negative slope test case."
        )

    # TODO: Add more sophisticated unit tests


if __name__ == "__main__":
    unittest.main()
