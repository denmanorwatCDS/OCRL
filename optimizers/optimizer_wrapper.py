import numpy as np


class DictBatchDataset:
    """Use when the input is the dict type."""
    def __init__(self, inputs, batch_size):
        self._inputs = inputs
        self._batch_size = batch_size
        self._size = list(self._inputs.values())[0].shape[0]
        if batch_size is not None:
            self._ids = np.arange(self._size)
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._size * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield self._inputs
        else:
            if update:
                self.update()
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = {
                    k: v[batch_ids]
                    for k, v in self._inputs.items()
                }
                yield batch

    def update(self):
        np.random.shuffle(self._ids)

class OptimizerGroupWrapper:
    """A wrapper class to handle torch.optim.optimizer.
    """

    def __init__(self,
                 optimizers,
                 max_optimization_epochs=1,
                 minibatch_size=None):
        self._optimizers = optimizers
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def get_minibatch(self, data, max_optimization_epochs=None):
        batch_dataset = DictBatchDataset(data, self._minibatch_size)

        if max_optimization_epochs is None:
            max_optimization_epochs = self._max_optimization_epochs

        for _ in range(max_optimization_epochs):
            for dataset in batch_dataset.iterate():
                yield dataset

    def zero_grad(self, keys=None):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        # TODO: optimize to param = None style.
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].zero_grad()

    def step(self, keys=None, **closure):
        """Performs a single optimization step.

        Arguments:
            **closure (callable, optional): A closure that reevaluates the
                model and returns the loss.

        """
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].step(**closure)

    def target_parameters(self, keys=None):
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            for pg in self._optimizers[key].param_groups:
                for p in pg['params']:
                    yield p