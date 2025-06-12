"""
odc.loader._driver
==================

This module manages reader drivers for `odc-loader`. It provides
functionality to register, unregister, and look up `ReaderDriver`
instances by name or specification string. Drivers are responsible for
opening and reading data from various raster formats.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from ._rio import RioDriver
from ._zarr import XrMemReaderDriver
from .types import ReaderDriver, ReaderDriverSpec, is_reader_driver

_available_drivers: dict[str, Callable[[], ReaderDriver] | ReaderDriver] = {
    "rio": RioDriver,
    "zarr": XrMemReaderDriver,
}


def register_driver(
    name: str, driver: Callable[[], ReaderDriver] | ReaderDriver, /
) -> None:
    """
    Register a new driver
    """
    _available_drivers[name] = driver


def unregister_driver(name: str, /) -> None:
    """
    Unregister a driver
    """
    _available_drivers.pop(name, None)


def _norm_driver(drv: Any) -> ReaderDriver:
    """
    Normalize a driver input to a ReaderDriver instance.

    If `drv` is a type, it's instantiated. If it's already a ReaderDriver,
    it's returned as is. Otherwise, it's assumed to be a callable that
    returns a ReaderDriver instance.

    :param drv: The driver object or type to normalize.
    :return: A ReaderDriver instance.
    """
    if isinstance(drv, type):
        return drv()
    if is_reader_driver(drv):
        return drv
    return drv()


def reader_driver(x: ReaderDriverSpec | None = None, /) -> ReaderDriver:
    """
    Resolve a ReaderDriverSpec to a ReaderDriver instance.

    - If `x` is None, returns the default RioDriver.
    - If `x` is already a ReaderDriver instance, returns `x`.
    - If `x` is a string:
        - Looks up registered drivers by name (e.g., "rio", "zarr").
        - If not found and `x` contains '.', treats it as "module.ClassName"
          and attempts to import and instantiate it.
    - Raises ValueError if the driver specification cannot be resolved.

    :param x: A ReaderDriverSpec (string name, module.Class string, or instance)
              or None to get the default driver.
    :return: A ReaderDriver instance.
    :raises ValueError: If the driver name or spec is unknown or cannot be resolved.
    """
    if x is None:
        return RioDriver()
    if not isinstance(x, str):
        return x

    if (drv := _available_drivers.get(x)) is not None:
        return _norm_driver(drv)

    if "." not in x:
        raise ValueError(f"Unknown driver: {x!r}")

    module_name, class_name = x.rsplit(".", 1)
    try:
        cls = getattr(import_module(module_name), class_name)
        return cls()
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Failed to resolve driver spec: {x!r}") from None
