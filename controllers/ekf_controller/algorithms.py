"""
Sensor based path finding algorithms, used with uncalibrated sensors and no reference to a global map 
reference.

These algorithms are purely reactive, accepting the eight epuck sensor readings as input and outputting left 
and right wheel velocities.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

class ReactiveEpuck(ABC):

    def __init__(self, base_velocity: float, distance_threshold: float):
        self.base_velocity = base_velocity
        self.distance_threshold = distance_threshold

    @abstractmethod
    def get_wheel_velocities(self, ps: Sequence[float]) -> Tuple[float, float]:
        """
        Compute left and right wheel velocities given proximity sensor readings.

        Args:
            ps: Sequence of 8 epuck sensor values.

        Returns:
            (lwv, rwv)
        """
        return 0.0, 0.0

class RightWallHugger(ReactiveEpuck):
    def __init__(self, base_velocity: float, distance_threshold: float):
        super().__init__(base_velocity, distance_threshold)
        return
    
    def get_wheel_velocities(self, ps):
        lwv, rwv = 0.0, 0.0
        if ps[0] > self.distance_threshold: # sharp left turn
            lwv, rwv = -self.base_velocity, self.base_velocity
        elif ps[1] > self.distance_threshold: # left adjustment
            lwv, rwv = self.base_velocity * 0.05, self.base_velocity
        elif ps[2] < self.distance_threshold: # right adjustment
            lwv, rwv = self.base_velocity, self.base_velocity * 0.05
        else: # drive forward
            lwv = rwv = self.base_velocity

        return lwv, rwv
    

class LeftWallHugger(ReactiveEpuck):
    def __init__(self, base_velocity: float, distance_threshold: float):
        super().__init__(base_velocity, distance_threshold)
        return
    
    def get_wheel_velocities(self, ps):
        lwv, rwv = 0.0, 0.0
        if ps[7] > self.distance_threshold: # sharp left right
            lwv, rwv = self.base_velocity, -self.base_velocity
        elif ps[6] > self.distance_threshold: # right adjustment
            lwv, rwv = self.base_velocity, self.base_velocity * 0.05
        elif ps[5] < self.distance_threshold: # left adjustment
            lwv, rwv = self.base_velocity * 0.05, self.base_velocity
        else: # drive forward
            lwv = rwv = self.base_velocity

        return lwv, rwv