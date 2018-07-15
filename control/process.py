import numpy as np
import math
from abc import ABC, abstractmethod


class Process(ABC):
    """Abstract class for a process to be controlled."""

    def __init__(self):
        self.y = np.asarray([0])
        self.x = np.asarray([0])
        self.e = np.asarray([0])
        self.u = np.asarray([0])
        self.x_aux = np.asarray([0])
        self.t = np.asarray([0])

        self.reset()

    @abstractmethod
    def target(self, t):
        pass

    @abstractmethod
    def sense(self, t):
        pass

    @abstractmethod
    def sense_aux(self, t):
        pass

    @abstractmethod
    def correct(self, e, dt):
        pass

    @abstractmethod
    def actuate(self, u, dt):
        pass

    @abstractmethod
    def distort(self, t, dt):
        pass

    def update(self, dt):
        self.distort(self.t, dt)

        # Read the set point
        self.y = self.target(self.t)
        # Measure the process variable
        self.x = self.sense(self.t)
        # Measure additional variables (not used for PID, just for monitoring)
        self.x_aux = self.sense_aux(self.t)
        # Compute error
        self.e = self.y - self.x
        # Perform correction based on error
        self.u = self.correct(self.e, dt)
        # Use the correction to act on process
        self.actuate(self.u, dt)

        self.t += dt

        return self.e

    def reset(self):
        self.t = 0.

    def loop(self, tsim=10, dt=0.01):
        """Loop the process until simulation time is reached.

        Returns a structured array of intermediate results for each iteration.
        """
    
        n = int(math.ceil(tsim / dt))

        for i in range(n):
            self.update(dt)

            if i == 0:
                fields = [
                    ('y', np.float, self.y.shape),
                    ('x', np.float, self.x.shape),
                    ('e', np.float, self.e.shape),
                    ('u', np.float, self.u.shape),
                    ('x_aux', np.float, self.x_aux.shape),
                    ('t', np.float, 1)
                ]
                result = np.zeros(n, dtype=fields)

            result[i] = (self.y, self.x, self.e, self.u, self.x_aux, self.t - dt)

        return result
