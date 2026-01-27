from typing import Self, Iterable, Union
import weakref
import numpy as np

from ..wrappers.decimal import Decimal as _decimal


class Point:
    def __array_ufunc__(self, func, method, inputs, instance, **kwargs):
        if func == np.matmul:
            if isinstance(instance, np.ndarray):
                arr = self.as_numpy
                arr @= instance
                x, y, z = arr

                self._x = _decimal(x)
                self._y = _decimal(y)
                self._z = _decimal(z)

                self._process_update()
                return self
            else:
                return inputs @ self.as_numpy

        if func == np.add:
            arr = self.as_numpy

            if isinstance(instance, np.ndarray):
                arr += instance
                x, y, z = arr
                self._x = _decimal(x)
                self._y = _decimal(y)
                self._z = _decimal(z)

                self._process_update()
                return self
            else:
                return inputs + arr

        if func == np.subtract:
            arr = self.as_numpy

            if isinstance(instance, np.ndarray):
                arr -= instance
                x, y, z = arr
                self._x = _decimal(x)
                self._y = _decimal(y)
                self._z = _decimal(z)

                self._process_update()
                return self
            else:
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

        raise RuntimeError

    def __init__(self, x: float, y: float, z: float | None = None):

        if z is None:
            self.is2d = True
            z = 0.0
        else:
            self.is2d = False

        self._x = _decimal(x)
        self._y = _decimal(y)
        self._z = _decimal(z)

        self._callbacks = []
        self._ref_count = 0

    def __enter__(self):
        self._ref_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ref_count -= 1

    def __remove_callback(self, ref):
        try:
            self._callbacks.remove(ref)
        except:  # NOQA
            pass

    def bind(self, callback):
        # we don't explicitly check to see if a callback is already registered
        # what we care about is if a callback is called only one time and that
        # check is done when the callbacks are being done and if there happend
        # to be a duplicate the duplicate is then removed at that point in time.
        ref = weakref.WeakMethod(callback, self.__remove_callback)

        self._callbacks.append(ref)

    def unbind(self, callback):
        for ref in self._callbacks[:]:
            cb = ref()
            if cb is None:
                self._callbacks.remove(ref)
            elif cb == callback:
                # we don't return after licating a matching callback in the
                # event a callback was registered more than one time. duplicates
                # are also removed aty the time callbacks get called but if an update
                # to a point never occurs we want to make sure that we explicitly
                # unbind all callbacks including duplicates.
                self._callbacks.remove(ref)

    def _process_update(self):
        if self._ref_count:
            return

        used_callbacks = []
        for ref in self._callbacks[:]:
            cb = ref()
            if cb is None:
                self._callbacks.remove(ref)
            elif cb not in used_callbacks:
                cb(self)
                used_callbacks.append(cb)
            else:
                # remove duplicate callbacks since we are
                # iterating over the callbacks
                self._callbacks.remove(ref)

    @property
    def x(self) -> float:
        return float(self._x)

    @x.setter
    def x(self, value: float):
        self._x = _decimal(value)
        self._process_update()

    @property
    def y(self) -> float:
        return float(self._y)

    @y.setter
    def y(self, value: float):
        self._y = _decimal(value)
        self._process_update()

    @property
    def z(self) -> float:
        return float(self._z)

    @z.setter
    def z(self, value: float):
        self._z = _decimal(value)
        self._process_update()

    def copy(self) -> "Point":
        return Point(self.x, self.y, self.z)

    def __iadd__(self, other: Union["Point", np.ndarray]) -> Self:
        if isinstance(other, Point):
            x, y, z = other
        else:
            x, y, z = (float(item) for item in other)

        self._x += _decimal(x)
        self._y += _decimal(y)
        self._z += _decimal(z)

        self._process_update()

        return self

    def __add__(self, other: Union["Point", np.ndarray]) -> "Point":
        x1, y1, z1 = self.as_decimal
        if isinstance(other, Point):
            x2, y2, z2 = other.as_decimal
        else:
            x2, y2, z2 = (_decimal(float(item)) for item in other)

        x = x1 + x2
        y = y1 + y2
        z = z1 + z2

        return Point(float(x), float(y), float(z))

    def __isub__(self, other: Union["Point", np.ndarray]) -> Self:
        if isinstance(other, Point):
            x, y, z = other.as_decimal
        else:
            x, y, z = (_decimal(float(item)) for item in other)

        self._x -= x
        self._y -= y
        self._z -= z

        self._process_update()

        return self

    def __sub__(self, other: Union["Point", np.ndarray]) -> "Point":
        x1, y1, z1 = self.as_decimal
        if isinstance(other, Point):
            x2, y2, z2 = other.as_decimal
        else:
            x2, y2, z2 = (_decimal(float(item)) for item in other)

        x = x1 - x2
        y = y1 - y2
        z = z1 - z2

        return Point(float(x), float(y), float(z))

    def __imul__(self, other: float) -> Self:
        other = _decimal(other)

        self._x *= other
        self._y *= other
        self._z *= other

        self._process_update()

        return self

    def __itruediv__(self, other: float) -> Self:
        other = _decimal(other)

        self._x /= other
        self._y /= other
        self._z /= other

        self._process_update()

        return self

    def __truediv__(self, other: float) -> "Point":
        other = _decimal(other)

        x, y, z = self.as_decimal
        return Point(float(x / other), float(y / other), float(z / other))

    def set_angle(self, angle: "_angle.Angle", origin: "Point"):
        arr = self.as_numpy

        arr -= origin
        arr @= angle.as_matrix
        arr += origin

        self._x = _decimal(arr[0])
        self._y = _decimal(arr[1])
        self._z = _decimal(arr[2])
        self._process_update()

    def get_angle(self, origin: "Point") -> "Angle":
        return _angle.Angle.from_points(origin, self)

    def __bool__(self):
        return tuple(self) == (0, 0, 0)

    def __eq__(self, other: "Point") -> bool:
        x1, y1, z1 = self
        x2, y2, z2 = other

        return x1 == x2 and y1 == y2 and z1 == z2

    def __ne__(self, other: "Point") -> bool:
        return not self.__eq__(other)

    @property
    def as_decimal(self) -> tuple[float, float, float]:
        return float(self._x), float(self._y), float(self._z)

    @property
    def as_int(self) -> tuple[int, int, int]:
        return int(self._x), int(self._y), int(self._z)

    @property
    def as_numpy(self) -> np.ndarray:
        return np.array(list(self), dtype=np.dtypes.Float64DType)

    def __iter__(self) -> Iterable[_decimal]:
        return iter([self.x, self.y, self.z])

    def __str__(self) -> str:
        return f'X: {self.x}, Y: {self.y}, Z: {self.z}'

    def __le__(self, other: "Point") -> bool:
        x1, y1, z1 = self
        x2, y2, z2 = other
        return x1 <= x2 and y1 <= y2 and z1 <= z2

    def __ge__(self, other: "Point") -> bool:
        x1, y1, z1 = self
        x2, y2, z2 = other
        return x1 >= x2 and y1 >= y2 and z1 >= z2

    @property
    def inverse(self) -> "Point":
        point = self.copy()

        point._x = -point._x
        point._y = -point._y
        point._z = -point._z

        return point


ZERO_POINT = Point(0.0, 0.0, 0.0)

from . import angle as _angle  # NOQA