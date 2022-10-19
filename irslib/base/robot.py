"""Body module.

The robot class abstracts a manipulator robot.

Example:
   >>> from irslib.base.robot import Robot

Todo:
    * IK(pos, pose), FK, jacobian
    * Documentation

"""
from .body import Body
from .pose import Pose
from typing import *
import numpy as np

class Robot(Body):
    def __init__(
        self,
        uid: int,
        ee_index: int,
        name:str = None
    ) -> "Robot":
        if name is None:
            name = f"robot{uid}"
        super().__init__(uid=uid, name=name)
        self.ee_index = ee_index

    def get_ee_pose(self) -> Pose:
        return self.get_link_pose(self.ee_index)
    
    def IK_pos(self, pos: np.ndarray):
        pass

    def IK_pose(self, pose: Pose, tol: float=1e-3, max_iter:int=100):
        pass
    
    def FK(self, joint_angles: np.ndarray):
        pass

    def get_jacobian(self, joint_angles: np.ndarray):
        pass