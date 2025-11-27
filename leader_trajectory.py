import numpy as np

class LeaderTrajectory:
    def __init__(self, trajectory_type="circle", center=None, radius=100, speed=1.0, axis="z"):
        self.trajectory_type = trajectory_type
        self.center = np.array(center) if center is not None else np.array([250, 250, 250])
        self.radius = radius
        self.speed = speed
        self.axis = axis

    def get_position(self, t, leader_idx=0, total_leaders=1):
        if self.trajectory_type == "circle":
            # Each leader offset by phase
            phase = 2 * np.pi * leader_idx / total_leaders
            angle = self.speed * t + phase
            if self.axis == "z":
                x = self.center[0] + self.radius * np.cos(angle)
                y = self.center[1] + self.radius * np.sin(angle)
                z = self.center[2]
            elif self.axis == "y":
                x = self.center[0] + self.radius * np.cos(angle)
                y = self.center[1]
                z = self.center[2] + self.radius * np.sin(angle)
            elif self.axis == "x":
                x = self.center[0]
                y = self.center[1] + self.radius * np.cos(angle)
                z = self.center[2] + self.radius * np.sin(angle)
            else:
                raise ValueError("Invalid axis for circle trajectory")
            return np.array([x, y, z])
        # Add more trajectory types as needed
        else:
            raise NotImplementedError(f"Trajectory type {self.trajectory_type} not implemented.")
