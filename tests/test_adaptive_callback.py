import os

os.environ["KERAS_BACKEND"] = "torch"
import unittest
import numpy as np
from keras import backend as K, ops
from softadapt.algorithms import (
    LossWeightedSoftAdapt,
    NormalizedSoftAdapt,
    SoftAdapt,
)

# Assuming AdaptiveLossCallback is defined in a module named adaptive_loss_callback
from softadapt.callbacks import AdaptiveLossCallback

class TestAdaptiveLossCallback(unittest.TestCase):

    def setUp(self):
        # Set up the initial parameters for the callback
        self.components = ['loss1', 'loss2']
        self.weights = [0.5, 0.5]
        self.callback = AdaptiveLossCallback(components=self.components, weights=self.weights, frequency='epoch', beta=0.1, algorithm='base')

    def test_initialization(self):
        # Test if the callback initializes correctly
        self.assertEqual(self.callback.order, self.components)
        self.assertTrue(isinstance(self.callback.algorithm, SoftAdapt))
        self.assertTrue(np.array_equal(ops.convert_to_numpy(self.callback._weights[0]), self.weights[0]))
        self.assertTrue(np.array_equal(ops.convert_to_numpy(self.callback._weights[1]), self.weights[1]))
        self.assertEqual(self.callback.frequency, 'epoch')
        self.assertEqual(len(self.callback.components_history), len(self.components))

    def test_on_epoch_end_updates_weights(self):
        # Simulate logs for the end of an epoch
        logs = {'loss1': 0.2, 'loss2': 0.3}
        self.callback.on_epoch_end(epoch=0, logs=logs)

        # Check if the component history is updated
        self.assertEqual(self.callback.components_history[0], [0.2])
        self.assertEqual(self.callback.components_history[1], [0.3])

        # Check if weights are updated (mocking the algorithm's behavior)
        # Here we would need to mock the get_component_weights method
        # For simplicity, let's assume it returns [0.6, 0.4]
        self.callback.algorithm.get_component_weights = lambda *x, verbose: np.array([0.6, 0.4])
        self.callback.on_epoch_end(epoch=1, logs=logs)

        # Check if the weights have been updated
        self.assertTrue(np.allclose(ops.convert_to_numpy(self.callback._weights), [0.6, 0.4]))

    def test_true_epoch_end_updates_weights(self):
        # Simulate logs for the end of an epoch
        logs = {'loss1': 0.2, 'loss2': 0.3}
        self.callback.on_epoch_end(epoch=0, logs=logs)

        # Check if the component history is updated
        self.assertEqual(self.callback.components_history[0], [0.2])
        self.assertEqual(self.callback.components_history[1], [0.3])

        # Check if weights are updated (mocking the algorithm's behavior)
        self.callback.on_epoch_end(epoch=1, logs=logs)

        # Check if the weights have been updated
        self.assertTrue(np.allclose(ops.convert_to_numpy(self.callback._weights), [0.5, 0.5]))

    def test_on_epoch_end_clears_history(self):
        # Simulate logs for the end of an epoch
        logs = {'loss1': 0.2, 'loss2': 0.3}
        self.callback.on_epoch_end(epoch=0, logs=logs)

        # Check if the component history is updated
        self.assertEqual(self.callback.components_history[0], [0.2])
        self.assertEqual(self.callback.components_history[1], [0.3])

        # Call on_epoch_end again to trigger clearing of history
        self.callback.algorithm.get_component_weights = lambda *x, verbose: np.array([0.6, 0.4])
        logs = {'loss1': 0.5, 'loss2': 0.2}
        self.callback.on_epoch_end(epoch=1, logs=logs)

        # Check if the history has been cleared except for one
        self.assertEqual(self.callback.components_history[0], [0.5])
        self.assertEqual(self.callback.components_history[1], [0.2])

    def tearDown(self):
        # Clean up any resources if needed
        K.clear_session()

if __name__ == '__main__':
    unittest.main()
