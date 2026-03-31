import numpy as np
from scipy.stats import qmc

class BaseSampler:
    """Base class for parameter sampling."""
    def __init__(self, d: int, seed: int = None):
        self.d = d
        self.seed = seed

    def random(self, n: int = 1):
        raise NotImplementedError

    def scale(self, sample, l_bounds, u_bounds):
        return qmc.scale(sample, l_bounds, u_bounds)

class RandomSampler(BaseSampler):
    """Wrapper for numpy.random.default_rng."""
    def __init__(self, d: int, seed: int = None):
        super().__init__(d, seed)
        self.rng = np.random.default_rng(seed)

    def random(self, n: int = 1):
        return self.rng.random((n, self.d))

class SobolSampler(BaseSampler):
    """Wrapper for scipy.stats.qmc.Sobol."""
    def __init__(self, d: int, scramble: bool = True, seed: int = None):
        super().__init__(d, seed)
        self.engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)

    def random(self, n: int = 1):
        # Sobol sequence is stateful.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*balance properties.*")
            return self.engine.random(n=n)
