"""Body module.

The body class abstracts a rigid body (or an articulated rigid bodies).

Example:
   >>> --
   

Todo:
    * Documentation

"""
from typing import *
import numpy as np
from abc import ABC
from dataclasses import dataclass, field

from .pose import Pose

init_none = lambda: None

@dataclass
class Joint:
    """Joint class
    """
    name: str = field(default_factory=init_none)
    joint_type: str = field(default_factory=init_none)
    lower_limit: float = field(default_factory=init_none)
    upper_limit: float = field(default_factory=init_none)
    max_force: float = field(default_factory=init_none)
    max_vel: float = field(default_factory=init_none)
    link_name: str = field(default_factory=init_none)
    parent_link_index: int = field(default_factory=init_none)
    child_link_index: int = field(default_factory=init_none)

class Body(ABC):
    def __init__(
        self,
        uid:int,
        name:str = None
    ) -> "Body":
        """_summary_

        Args:
            uid (int): unique id of the body

        Returns:
            Body: Body instance
        """
        self.uid = uid
        self.joints = None
        self.name = name if name is not None else f"body{uid}"
    
    @property
    def joint_lower_limit(self) -> np.ndarray:
        pass

    @property
    def joint_upper_limit(self) -> np.ndarray:
        pass
    
    @property
    def joint_mid(self) -> np.ndarray:
        pass
    
    def get_base_pose(self) -> Pose:
        pass

    def get_base_vel(self) -> np.ndarray:
        #linear, angular
        pass
    
    def get_link_pose(self, link_index:int)->Pose:
        pass

    def get_link_vel(self, link_index:int)->np.ndarray:
        pass
    
    def get_joint_angle(self, joint_index:int) -> np.ndarray:
        pass

    def get_joint_angles(self) -> np.ndarray:
        num_joints = len(self.joints)
        return np.array([self.get_joint_angle(i) for i in range(num_joints)])

    def get_joint_vel(self, joint_index: int) -> float:
        pass








    
