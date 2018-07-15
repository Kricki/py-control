import numpy as np


class PID:
    """Implementation of a PID controller.

    This implementation is geared towards discrete time systems,
    where PID is often called PSD (proportional-sum-difference).

    Usage is fairly straight forward. Set the coefficients of
    the three terms to values of your choice and call PID.update
    with constant timesteps.
    """
    
    def __init__(self, gain=1, kp=0.5, ki=0.0, kd=0.01):
        self.gain = gain
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_max = None
        self.output_min = None

        self._first = True
        self._last_error = 0
        self._sum_error = 0

        self.fc = 0
        self._last_pid_input = 0

    def update(self, error, dt):
        """Update the PID controller.
       
        Computes the new control value as                 
            u(t) = gain*(kp*err(t) + kd*d/dt(err(t)) + ki*I(e))
        
        where I(e) is the integral of the error up to the current timepoint.

        Args:
            error: Error between set point and measured value
            dt: Time step delta

        Returns:
            Returns the control value u(t)
        """

        if self._first:
            self._last_error = np.copy(error)
            self._sum_error = np.zeros(error.shape)
            self._first = False

        # High-pass filter for input error signal
        # fc is the cut-off frequency
        # Implementation according to https://en.wikipedia.org/wiki/High-pass_filter
        if self.fc > 0:
            alpha = 1/(1+2*np.pi*self.fc*dt)
            pid_input = alpha*(self._last_pid_input+error-self._last_error)
        else:
            pid_input = error

        self._last_error = error

        #self._sum_error += error*dt
        self._sum_error += pid_input * dt
        #derr = (error - self._last_error) / dt
        derr = (pid_input - self._last_pid_input) / dt

        self._last_pid_input = pid_input

        #u = (self.kp * error + self.kd * derr + self.ki * self._sum_error)*self.gain
        u = (self.kp * pid_input + self.kd * derr + self.ki * self._sum_error) * self.gain

        # clip control value to output limits
        if (self.output_min is not None) or (self.output_max is not None):
            u = np.clip(u, self.output_min, self.output_max)

        return u

    def reset(self):
        self._first = True
