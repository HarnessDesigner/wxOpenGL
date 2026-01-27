from typing import Self, Callable, Iterable, Union

import weakref
import math
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from ..wrappers.decimal import Decimal as _decimal
from . import point as _point


class Angle:
    """
    Makes is easier to set angles.

    This class defines several "dunder" (double underscore) methods or "magic"
    methods that allow for using math operators to perform rotation tasks.

    numpy arrays are able to be used on it and instances of thos class are
    able to be used on numpy arrays. Instances of this class are also able to be
    applied to an instance of the `wxOpenGL.geometry.point.Point` class as well.

    This class only deals with angles as X, Y, Z and only in degrees. it is also
    written to have the coordinate system done as

    -X: left
    +X: right
    -Y: down
    +Y: up
    -Z: near
    +Z: far
    """

    def __array_ufunc__(self, func, method, inputs, instance, **kwargs):
        """
        Private method for working with numpy arrays
        """
        if func == np.matmul:
            if isinstance(instance, np.ndarray):
                if instance.shape == (1, 3):
                    angle = self.from_euler(*instance.tolist())
                elif instance.shape == (1, 4):
                    angle = self.from_quat(instance)
                elif instance.shape == (3, 3):
                    angle = self.from_matrix(instance)
                else:
                    raise ValueError('array has an incorrect shape')

                angle = self @ angle
                self._R = angle._R  # NOQA
                self._process_update()
                return self
            else:
                return inputs @ self._R.as_matrix().T  # NOQA

        if func == np.add:
            if isinstance(instance, np.ndarray):
                if instance.shape == (1, 3):
                    angle = self.from_euler(*instance.tolist())
                elif instance.shape == (1, 4):
                    angle = self.from_quat(instance)
                elif instance.shape == (3, 3):
                    angle = self.from_matrix(instance)
                else:
                    raise ValueError('array has an incorrect shape')

                angle = self + angle
                self._R = angle._R  # NOQA
                self._process_update()
                return self
            else:
                arr = self.as_euler_array
                return inputs + arr

        if func == np.subtract:
            if isinstance(instance, np.ndarray):
                if instance.shape == (1, 3):
                    angle = self.from_euler(*instance.tolist())
                elif instance.shape == (1, 4):
                    angle = self.from_quat(instance)
                elif instance.shape == (3, 3):
                    angle = self.from_matrix(instance)
                else:
                    raise ValueError('array has an incorrect shape')

                angle = self - angle
                self._R = angle._R  # NOQA
                self._process_update()
                return self
            else:
                arr = self.as_euler_array
                return inputs + arr

        print('func:', func)
        print()
        print('method:', method)
        print()
        print('inputs:', inputs)
        print()
        print('instance:', instance)
        print()
        print('kwargs:', kwargs)
        print()
        raise RuntimeError('Please report this error to '
                           '`https://github.com/kdschlosser/wxOpenGL/issues`')

    def __init__(self, R=None):
        """
        Internal use. please use the
        `Angle.from_quat`, `Angle.from_points`, `Angle.from_euler` or
        `Angle.from_matrix`
        """

        if R is None:
            R = _Rotation.from_quat([0.0, 0.0, 0.0, 1.0])  # NOQA

        self._R = R

        self.__callbacks = []
        self._ref_count = 0

    def __enter__(self):
        self._ref_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ref_count -= 1

    def __remove_callback(self, ref):
        """
        Internal use

        :param ref: weakref.rf instance
        :return:
        """
        try:
            self.__callbacks.remove(ref)
        except:  # NOQA
            pass

    def bind(self, cb: Callable[["Angle"], None]):
        """
        Bind a callback method to be called if the angle changes.

        ..note: Only methods can be supplied, There is no support for functions.

        :param cb: callback
        """
        # We don't explicitly check to see if a callback is already registered.
        # What we care about is if a callback is called only one time and that
        # check is done when the callbacks are being executed. If there happens
        # to be a duplicate, the duplicate is then removed at that point in time.
        ref = weakref.WeakMethod(cb, self.__remove_callback)
        self.__callbacks.append(ref)

    def unbind(self, cb: Callable[["Angle"], None]) -> None:
        """
        Unbind a callback method.

        :param cb: Callback method that was bound.
        """
        for ref in self.__callbacks[:]:
            callback = ref()
            if callback is None:
                self.__callbacks.remove(ref)
            elif callback == cb:
                # We don't return after locating a matching callback in the
                # event a callback was registered more than one time. Duplicates
                # are also removed at the time callbacks get called but if an update
                # to an angle never occurs we want to make sure that we explicitly
                # unbind all callbacks including duplicates.
                self.__callbacks.remove(ref)

    def _process_update(self):
        """
        Internal use.
        """
        if self._ref_count:
            return

        for ref in self.__callbacks[:]:
            cb = ref()
            if cb is None:
                self.__callbacks.remove(ref)
            else:
                cb(self)

    @property
    def inverse(self) -> "Angle":
        """
        Get the inverse of the angle.

        :rtype: `Angle`
        """

        R = self._R.inv()
        return Angle(R)

    @classmethod
    def _quat_to_euler(cls, quat: np.ndarray) -> tuple[_decimal, _decimal, _decimal]:
        """
        Internal use.
        """
        c = quat[0]
        d = quat[1]
        e = quat[2]
        f = quat[3]
        g = c + c
        h = d + d
        k = e + e
        a = c * g
        l = c * h  # NOQA
        c = c * k
        m = d * h
        d = d * k
        e = e * k
        g = f * g
        h = f * h
        f = f * k

        matrix = np.array([1 - (m + e), l + f, c - h, 0,
                           l - f, 1 - (a + e), d + g, 0,
                           c + h, d - g, 1 - (a + m), 0], dtype=np.float64)

        return cls._matrix_to_euler(matrix)

    @staticmethod
    def _matrix_to_euler(matrix: np.ndarray) -> tuple[_decimal, _decimal, _decimal]:
        """
        Internal use.
        """
        def clamp(a_, b_, c_):
            return max(b_, min(c_, a_))

        a = matrix[0]
        f = matrix[4]
        g = matrix[8]
        h = matrix[1]
        k = matrix[5]
        l = matrix[9]  # NOQA
        m = matrix[2]
        n = matrix[6]
        e = matrix[10]

        y = math.asin(clamp(g, -1, 1))
        if 0.99999 > abs(g):
            x = math.atan2(-l, e)
            z = math.atan2(-f, a)
        else:
            x = math.atan2(n, k)
            z = 0

        return (_decimal(math.degrees(x)), _decimal(math.degrees(y)),
                _decimal(math.degrees(z)))

    @staticmethod
    def _euler_to_quat(x: _decimal, y: _decimal, z: _decimal) -> np.ndarray:
        """
        Internal use.
        """
        x = _decimal(math.radians(float(x)))
        y = _decimal(math.radians(float(y)))
        z = _decimal(math.radians(float(z)))

        d2 = _decimal(2.0)
        x_half = x / d2
        y_half = y / d2
        z_half = z / d2

        c = math.cos(x_half)
        d = math.cos(y_half)
        e = math.cos(z_half)
        f = math.sin(x_half)
        g = math.sin(y_half)
        h = math.sin(z_half)

        x = f * d * e + c * g * h
        y = c * g * e - f * d * h
        z = c * d * h + f * g * e
        w = c * d * e - f * g * h

        quat = np.array([float(x), float(y), float(z), float(w)], dtype=np.float32)
        return quat

    @property
    def x(self) -> float:
        """
        Get the X euler angle in degrees.

        :rtype: float
        """
        quat = self._R.as_quat()
        angles = self._quat_to_euler(quat)
        return float(angles[0])

    @x.setter
    def x(self, value: float):
        """
        Set the X euler angle in degrees.

        :param value: X euler angle in degrees
        :type value: float
        """

        quat = self._R.as_quat()
        angles = list(self._quat_to_euler(quat))
        angles[0] = _decimal(value)
        quat = self._euler_to_quat(*angles)

        self._R = _Rotation.from_quat(quat)  # NOQA
        self._process_update()

    @property
    def y(self) -> float:
        """
        Get the Y euler angle in degrees.

        :rtype: float
        """
        quat = self._R.as_quat()
        angles = self._quat_to_euler(quat)
        return float(angles[1])

    @y.setter
    def y(self, value: float):
        """
        Set the Y euler angle in degrees.

        :param value: Y euler angle in degrees
        :type value: float
        """
        quat = self._R.as_quat()
        angles = list(self._quat_to_euler(quat))
        angles[1] = _decimal(value)

        quat = self._euler_to_quat(*angles)

        self._R = _Rotation.from_quat(quat)  # NOQA
        self._process_update()

    @property
    def z(self) -> float:
        """
        Get the Z euler angle in degrees.

        :rtype: float
        """
        quat = self._R.as_quat()
        angles = self._quat_to_euler(quat)
        return float(angles[2])

    @z.setter
    def z(self, value: float):
        """
        Set the Z euler angle in degrees.

        :param value: Z euler angle in degrees
        :type value: float
        """
        quat = self._R.as_quat()
        angles = list(self._quat_to_euler(quat))
        angles[2] = _decimal(value)

        quat = self._euler_to_quat(*angles)

        self._R = _Rotation.from_quat(quat)  # NOQA
        self._process_update()

    def copy(self) -> "Angle":
        return Angle.from_quat(self._R.as_quat())

    def __convert_other_to_quat(self, other):
        """
        Internal use.
        """

        if isinstance(other, Angle):
            quat = other.as_quat
        else:
            if isinstance(other, (list, tuple)):
                t_other = np.array(other, dtype=np.float64)
            elif isinstance(other, np.ndarray):
                t_other = other
            else:
                raise TypeError(f'type {type(other)} not supported')

            if t_other.shape() == (1, 3):
                quat = self._euler_to_quat(*(_decimal(item) for item in other.tolist()))
            elif t_other.shape() == (1, 4):
                quat = other
            elif t_other.shape() == (3, 3):
                quat = self._euler_to_quat(*self._matrix_to_euler(t_other))
            else:
                raise ValueError('incorrect shape for array')

        return quat

    def __iadd__(self, other: Union["Angle", np.ndarray, tuple, list]) -> Self:
        """
        Left hand addition.

        angle.Angle += other

        :param other: Amount to add to angle. <br/>

                      Allowed types:

                      * `angle.Angle` instance.
                      * numpy array, list or tuple of 3 floats `[x, y, z]` that are euler angles in degrees.
                      * numpy array, list or tuple of 4 floats `[x, y, z, w]` that is a quaternion.
                      * numpy array, list or tuple that has the shape of 3, 3 that is a rotation matrix.


        :type other: `Angle`, `np.ndarray`, `list` or `tuple`
        :return: self
        :rtype: self
        """

        new = self + other
        self._R = new._R  # NOQA

        self._process_update()
        return self

    def __add__(self, other: Union["Angle", np.ndarray, tuple, list]) -> "Angle":
        """
        Addition.

        new_angle = angle.Angle + other

        :param other: Amount to add to angle. <br/>

                      Allowed types:

                        * `angle.Angle` instance.
                        * numpy array, list or tuple of 3 floats `[x, y, z]` that are euler angles in degrees.
                        * numpy array, list or tuple of 4 floats `[x, y, z, w]` that is a quaternion.
                        * numpy array, list or tuple that has the shape of 3, 3 that is a rotation matrix.

        :type other: `Angle`, `np.ndarray`, `list` or `tuple`
        :return: new Angle instance
        :rtype: `Angle`
        """

        quat = self.__convert_other_to_quat(other)
        o_quat = self.as_quat
        o_quat += quat

        return self.from_quat(quat)

    def __isub__(self, other: Union["Angle", np.ndarray, tuple, list]) -> Self:
        """
        Left hand subtraction.

        angle.Angle -= other

        :param other: Amount to subtract from angle. <br/>

                      Allowed types:

                      * `angle.Angle` instance.
                      * numpy array, list or tuple of 3 floats `[x, y, z]` that are euler angles in degrees.
                      * numpy array, list or tuple of 4 floats `[x, y, z, w]` that is a quaternion.
                      * numpy array, list or tuple that has the shape of 3, 3 that is a rotation matrix.


        :type other: `Angle`, `np.ndarray`, `list` or `tuple`
        :return: self
        :rtype: self
        """
        new = self - other
        self._R = new._R  # NOQA

        self._process_update()
        return self

    def __sub__(self, other: Union["Angle", np.ndarray, tuple, list]) -> "Angle":
        """
        Subtraction.

        new_angle = angle.Angle - other

        :param other: Amount to subtract from angle. <br/>

                      Allowed types:

                        * `angle.Angle` instance.
                        * numpy array, list or tuple of 3 floats `[x, y, z]` that are euler angles in degrees.
                        * numpy array, list or tuple of 4 floats `[x, y, z, w]` that is a quaternion.
                        * numpy array, list or tuple that has the shape of 3, 3 that is a rotation matrix.

        :type other: `Angle`, `np.ndarray`, `list` or `tuple`
        :return: new Angle instance
        :rtype: `Angle`
        """
        quat = self.__convert_other_to_quat(other)
        o_quat = self.as_quat
        o_quat -= quat

        return self.from_quat(quat)

    def __rmatmul__(self, other: Union[np.ndarray, _point.Point]) -> np.ndarray:
        if isinstance(other, np.ndarray):
            other @= self._R.as_matrix().T
        elif isinstance(other, _point.Point):
            values = other.as_numpy @ self._R.as_matrix().T

            x = _decimal(float(values[0]))
            y = _decimal(float(values[1]))
            z = _decimal(float(values[2]))
            quat = self._euler_to_quat(x, y, z)
            other._R = _Rotation.from_quat(quat)  # NOQA
            other._process_update()  # NOQA

        elif isinstance(other, Angle):
            matrix = other._R.as_matrix() @ self._R.as_matrix()  # NOQA
            other._R = _Rotation.from_matrix(matrix)  # NOQA
            other._process_update()
        else:
            raise RuntimeError('sanity check')

        return other

    def __imatmul__(self, other: Union["Angle", np.ndarray, _point.Point]) -> np.ndarray | Self:
        if isinstance(other, np.ndarray):
            other @= self._R.as_matrix().T
        elif isinstance(other, _point.Point):
            values = other.as_numpy @ self._R.as_matrix().T

            x = _decimal(float(values[0]))
            y = _decimal(float(values[1]))
            z = _decimal(float(values[2]))

            with other:
                other.x = x
                other.y = y
                other.z = z

            other._process_update()  # NOQA

        elif isinstance(other, Angle):
            matrix = self._R.as_matrix() @ other._R.as_matrix()  # NOQA
            self._R = _Rotation.from_matrix(matrix)  # NOQA
            self._process_update()
            return self
        else:
            raise RuntimeError('sanity check')

        return other

    def __imul__(self, other):
        new = self * other
        self._R = new._R
        self._process_update()
        return self

    def __mul__(self, other):

        quat = self.__convert_other_to_quat(other)
        angle = self.from_quat(quat)

        return Angle(self._R * angle._R)  # NOQA

    def __matmul__(self, other: Union[_point.Point, np.ndarray]) -> np.ndarray:
        """
        Apply rotation.
        """

        if isinstance(other, np.ndarray):
            other = other @ self._R.as_matrix().T
        elif isinstance(other, _point.Point):
            values = other.as_numpy @ self._R.as_matrix().T
            x = _decimal(float(values[0]))
            y = _decimal(float(values[1]))
            z = _decimal(float(values[2]))

            other = _point.Point(x, y, z)
        elif isinstance(other, Angle):
            matrix = self._R.as_matrix() @ other._R.as_matrix()  # NOQA
            R = _Rotation.from_matrix(matrix)  # NOQA
            other = Angle(R)
        else:
            raise RuntimeError('sanity check')

        return other

    def __bool__(self):
        return tuple(self) == (0, 0, 0)

    def __eq__(self, other: "Angle") -> bool:
        x1, y1, z1 = self.as_decimal
        x2, y2, z2 = other.as_decimal

        return x1 == x2 and y1 == y2 and z1 == z2

    def __ne__(self, other: "Angle") -> bool:
        return not self.__eq__(other)

    @property
    def as_euler_array(self) -> np.ndarray:
        return np.array(list(self), dtype=np.float64)

    @property
    def as_decimal(self) -> tuple[float, float, float]:
        quat = self._R.as_quat()
        x, y, z = self._quat_to_euler(quat)

        return float(x), float(y), float(z)

    @property
    def as_int(self) -> tuple[int, int, int]:
        x, y, z = self
        return int(x), int(y), int(z)

    @property
    def as_quat(self) -> np.ndarray:
        return self._R.as_quat()

    @property
    def as_matrix(self) -> np.ndarray:
        return self._R.as_matrix().T  # NOQA

    def __iter__(self) -> Iterable[_decimal]:
        quat = self._R.as_quat()
        x, y, z = self._quat_to_euler(quat)

        return iter([float(x), float(y), float(z)])

    def __str__(self) -> str:
        quat = self._R.as_quat()
        x, y, z = self._quat_to_euler(quat)
        return f'X: {x}, Y: {y}, Z: {z}'

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        R = _Rotation.from_matrix(matrix)  # NOQA
        return cls(R)

    @classmethod
    def from_euler(cls, x: float | _decimal, y: float | _decimal,
                   z: float | _decimal) -> "Angle":

        quat = cls._euler_to_quat(x, y, z)
        R = _Rotation.from_quat(quat)  # NOQA
        ret = cls(R)
        return ret

    @classmethod
    def from_quat(cls, q: list[float, float, float, float] | np.ndarray) -> "Angle":

        R = _Rotation.from_quat(q)  # NOQA
        return cls(R)

    @classmethod
    def from_points(cls, p1: _point.Point, p2: _point.Point) -> "Angle":  # NOQA

        # the sign for all of the verticies in the array needs to be flipped in
        # order to handle the -Z axis being near
        p1 = -p1.as_numpy
        p2 = -p2.as_numpy

        f = p2 - p1

        fn = np.linalg.norm(f)
        if fn < 1e-6:
            return cls.from_euler(0.0, 0.0, 0.0)

        f = f / fn  # world-space direction of the line

        local_forward = np.array([0.0, 0.0, -1.0],
                                 dtype=np.dtypes.Float64DType)
        nz = np.nonzero(local_forward)[0][0]
        sign = np.sign(local_forward[nz])
        forward_world = f * sign

        up = np.asarray((0.0, 1.0, 0.0),
                        dtype=np.dtypes.Float64DType)

        if np.allclose(np.abs(np.dot(forward_world, up)), 1.0, atol=1e-8):
            up = np.array([0.0, 0.0, 1.0],
                          dtype=np.dtypes.Float64DType)

            if np.allclose(np.abs(np.dot(forward_world, up)), 1.0, atol=1e-8):
                up = np.array([1.0, 0.0, 0.0],
                              dtype=np.dtypes.Float64DType)

        right = np.cross(up, forward_world)  # NOQA

        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
            # raise RuntimeError("degenerate right vector")
        else:
            right = right / rn

        true_up = np.cross(forward_world, right)  # NOQA

        rot = np.column_stack((right, true_up, forward_world))

        R = _Rotation.from_matrix(rot)  # NOQA
        return cls(R)
