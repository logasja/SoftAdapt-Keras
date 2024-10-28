from typing import Literal
from keras import callbacks, ops, backend as K

from softadapt.algorithms import (
    LossWeightedSoftAdapt,
    NormalizedSoftAdapt,
    SoftAdapt,
)


class AdaptiveLossCallback(callbacks.Callback):
    def __init__(
        self,
        components: list[str],
        weights: list[float],
        frequency: Literal["epoch"] | Literal["batch"] | int = "epoch",
        beta: float = 0.1,
        accuracy_order: int = None,
        algorithm: Literal["loss-weighted"]
        | Literal["normalized"]
        | Literal["base"] = "base",
        calculate_on_validation=False,
    ):
        if algorithm == "base":
            self.algorithm = SoftAdapt(beta=beta, accuracy_order=accuracy_order)
        elif algorithm == "loss-weighted":
            self.algorithm = LossWeightedSoftAdapt(
                beta=beta, accuracy_order=accuracy_order
            )
        else:
            self.algorithm = NormalizedSoftAdapt(
                beta=beta, accuracy_order=accuracy_order
            )

        self.frequency = frequency
        self.order = components
        self._weights = weights
        self.components_history = [[] for _ in components]
        self.debug = False
        self.val = calculate_on_validation

    @property
    def weights(self) -> list[float]:
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def on_epoch_end(self, epoch, logs=None):
        # Update component history in order for weight computation
        if self.val:
            for k in self.order:
                self.components_history[self.order.index(k)].append(
                    ops.copy(logs["val_" + k])
                )
        else:
            for k in self.order:
                self.components_history[self.order.index(k)].append(ops.copy(logs[k]))

        # If the set number of epochs or frequency is met than recompute loss weights
        if (self.frequency == "epoch" or epoch % self.frequency == 0) and epoch != 0:
            adapt_weights = self.algorithm.get_component_weights(
                *ops.convert_to_tensor(self.components_history), verbose=self.debug
            )

            self.weights = ops.cast(adapt_weights, K.floatx())

            for h in self.components_history:
                if (
                    self.frequency == "epoch"
                ):  # In the case of an epoch-wise evaluation, the most recent loss value is retained
                    h.pop(0)
                else:
                    h.clear()
