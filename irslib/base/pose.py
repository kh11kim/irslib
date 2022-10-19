# -*- coding: utf-8 -*-
"""Pose(SE3) and Rotation(SO3) module.

This module contains two classes, ``Rotation`` and  ``Pose`` for
calculating spatial math.

Example:
   >>> from roboticstoolbox import ET
   >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
   

Todo:
    * Documentation

"""
import numpy as np
from typing import *
import scipy.spatial.transform


class Rotation(scipy.spatial.transform.Rotation):
    """ Wrapper class of ``scipy.spatial.transform.Rotation`` class (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html).
    
    You can make an instance using classmethods as below.

        >>> I = Rotation.identity(cls)
        >>> rot1 = Rotation.from_qtn(qtn)
        >>> rot2 = Rotation.from_matrix(rot_matrix_33)
        >>> rot3 = Rotation.from_euler("xyz", rpy)
    
    You can also convert one into other representations.

        >>> qtn = Rotation.as_qtn()
        >>> rot_matrix_33 = Rotation.as_matrix()
        >>> rpy = Rotation.as_euler("zyx")
    """

    @classmethod
    def identity(cls) -> "Rotation":
        """Identity rotation

        Returns:
            Rotation: rotation
        """
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])
        
    @classmethod
    def from_qtn(cls, qtn:np.ndarray) -> "Rotation":
        """A wrapper function of ``Rotation.from_quat()`` for naming consistency.

        Args:
            qtn (np.ndarray): a quaternion

        Returns:
            Rotation: Rotation instance.
        """
        return cls.from_quat(qtn)

    @classmethod
    def as_qtn(cls) -> np.ndarray:
        """A wrapper function of ``Rotation.as_quat()`` for naming consistency.

        Returns:
            qtn (np.ndarray): quaternion array
        """
        return cls.as_quat()

    def inverse(self) -> "Rotation":
        """Inverse operation

        Returns:
            Rotation: Rotation instance
        """
        return self.inv()

    def angle_between(self, rot: "Rotation") -> float:
        """Derive the angle between two rotation objects

        Args:
            rot (Rotation): rotation

        Returns:
            float: the angle in radian
        """
        error = self.inverse() * rot
        return error.magnitude()

    def slerp(self, other: "Rotation", num_interp: int) -> List["Rotation"]:
        """Slerp interpolation using two rotations

        Args:
            rot (Rotation): rotation
            num_interp (int): the number of intermediate points including the end point

        Returns:
            result (List[Rotation]): interpolated rotations
        """
        rots = Rotation.concatenate([self, other])
        ratio = np.linspace(0, 1, num_interp, endpoint=True)
        interpolator = scipy.spatial.transform.Slerp([0, 1], rots)
        interp_rots = interpolator(ratio)
        result = [Rotation.from_scipy(rot) for rot in interp_rots]
        return result
    
    @classmethod
    def from_scipy(self, rot:scipy.spatial.transform.Rotation) -> "Rotation":
        qtn = rot.as_quat()
        return Rotation.from_qtn(qtn)
        

class Pose:
    """Rigid spatial transform between coordinate systems in 3D space.

    You can make an instance using classmethods as below.

        >>> I = Pose.identity(cls)
        >>> rot, trans = Rotation.identity(), [1, 0, 3]
        >>> pose1 = Pose(rot, trans)
        >>> pose2 = Pose.from_matrix(pose_mat_44)
    """

    def __init__(
            self,
            rot: Rotation = Rotation.identity(),
            trans: Union[np.ndarray, list] = [0., 0., 0.]):
        assert isinstance(rot, scipy.spatial.transform.Rotation)
        assert isinstance(trans, (np.ndarray, list, tuple))

        self.rot = rot
        self.trans = np.asarray(trans, np.double)

    def __eq__(self, other: "Pose"):
        same_rot = np.allclose(self.rot.as_quat(), other.rot.as_quat())
        same_trans = np.allclose(self.trans, other.trans)
        return same_rot & same_trans

    def __repr__(self):
        trans = " ".join([f"{i:.3f}" for i in self.trans])
        rot = " ".join([f"{i:.3f}" for i in self.rot.as_quat()])
        return f"pose: trans[{trans}]-rot[{rot}]]"

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rot.as_matrix(), self.trans], [0.0, 0.0, 0.0, 1.0])
        )

    def as_1d_numpy(self):
        """Represet as a 1x7 vector [x,y,z,qx,qy,qz,qw]"""
        return np.hstack([*self.trans, *self.rot.as_quat()])

    def __mul__(self, other: "Pose") -> "Pose":
        """Compose this transformation with another."""
        rotation = self.rot * other.rot
        translation = self.rot.apply(other.trans) + self.trans
        return self.__class__(rotation, translation)

    def transform_point(self, point: Union[np.ndarray, list]):
        return self.rot.apply(point) + self.trans

    def transform_vector(self, vector: Union[np.ndarray, list]):
        return self.rot.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rot.inv()
        translation = -rotation.apply(self.trans)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()

# Quaternion functions
def qtn_conj(qtn):
    return np.hstack([-qtn[:3], qtn[-1]])


def qtn_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([x, y, z, w])


def orn_error(desired, current):
    cc = qtn_conj(current)
    q_r = qtn_mul(desired, cc)
    return q_r[:3] * np.sign(q_r[-1])


if __name__ == "__main__":
    p = Pose(Rotation.random(), np.random.random(3))
    r1 = Rotation.random()
    r2 = Rotation.random()
    result = r1.slerp(r2, 3)
    print("a")