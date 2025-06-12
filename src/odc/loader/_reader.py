"""
odc.loader._reader
==================

This module provides core utilities and classes for the raster reading pipeline.
It includes:
- `ReaderDaskAdaptor`: A class to adapt a standard `ReaderDriver` for use with Dask.
- Helper functions for resolving load configurations (`resolve_load_cfg`),
  nodata values (`resolve_src_nodata`, `resolve_dst_nodata`, `resolve_dst_fill_value`),
  data types (`resolve_dst_dtype`), and band selections (`resolve_band_query`).
- Utilities for handling overviews (`pick_overview`) and nodata comparisons/masks
  (`same_nodata`, `nodata_mask`).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from dask import delayed
from odc.geo.geobox import GeoBox

from .types import (
    Band_DType,
    RasterBandMetadata,
    RasterLoadParams,
    RasterSource,
    ReaderDriver,
    ReaderSubsetSelection,
    with_default,
)


def _dask_read_adaptor(
    src: RasterSource,
    ctx: Any,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    driver: ReaderDriver,
    env: dict[str, Any],
    selection: Optional[ReaderSubsetSelection] = None,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """
    Dask delayed task to perform a read operation using a given driver.

    This function is wrapped by `dask.delayed` to become part of a Dask graph.
    It restores the driver's environment, opens the source, and performs the read.

    :param src: The `RasterSource` to read from.
    :param ctx: The context for the reader driver's `restore_env` method.
    :param cfg: `RasterLoadParams` for the read operation.
    :param dst_geobox: The destination `GeoBox`.
    :param driver: The `ReaderDriver` instance.
    :param env: The environment dictionary for the driver's `restore_env` method.
    :param selection: Optional selection for subsetting the read.
    :return: A tuple containing the destination ROI (slices) and the numpy array of pixels.
    """
    with driver.restore_env(env, ctx) as local_ctx:
        rdr = driver.open(src, local_ctx)
        return rdr.read(cfg, dst_geobox, selection=selection)


class ReaderDaskAdaptor:
    """
    Creates default ``DaskRasterReader`` from a ``ReaderDriver``.

    Suitable for implementing ``.dask_reader`` property for generic reader drivers.
    """

    def __init__(
        self,
        driver: ReaderDriver,
        env: dict[str, Any] | None = None,
        ctx: Any | None = None,
        src: RasterSource | None = None,
        cfg: RasterLoadParams | None = None,
        layer_name: str = "",
        idx: int = -1,
    ) -> None:
        """
        Initialize ReaderDaskAdaptor.

        :param driver: The `ReaderDriver` to adapt.
        :param env: Optional environment dictionary for the driver. If None, it's captured from the driver.
        :param ctx: Optional context, usually provided when `open` is called.
        :param src: Optional `RasterSource`, usually provided when `open` is called.
        :param cfg: Optional `RasterLoadParams`, usually provided when `open` is called.
        :param layer_name: Name for Dask layers, used in task naming.
        :param idx: Index, often used for distinguishing tasks related to different sources.
        """
        if env is None:
            env = driver.capture_env()

        self._driver = driver
        self._env = env
        self._ctx = ctx
        self._src = src
        self._cfg = cfg
        self._layer_name = layer_name
        self._src_idx = idx

    def read(
        self,
        dst_geobox: GeoBox,
        *,
        selection: Optional[ReaderSubsetSelection] = None,
        idx: tuple[int, ...],
    ) -> Any:
        """
        Perform a Dask-delayed read operation.

        Constructs a Dask delayed task for reading data from the source
        configured in this adaptor.

        :param dst_geobox: The destination `GeoBox`.
        :param selection: Optional selection for subsetting the read.
        :param idx: Index tuple, used for Dask task naming and potentially by the reader.
        :return: A Dask delayed object representing the future result of the read.
        """
        assert self._src is not None
        assert self._ctx is not None
        assert self._cfg is not None

        read_op = delayed(_dask_read_adaptor, name=self._layer_name)

        # TODO: supply `dask_key_name=` that makes sense
        return read_op(
            self._src,
            self._ctx,
            self._cfg,
            dst_geobox,
            self._driver,
            self._env,
            selection=selection,
            dask_key_name=(self._layer_name, *idx),
        )

    def open(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        ctx: Any,
        layer_name: str,
        idx: int,
    ) -> "ReaderDaskAdaptor":
        """
        Configure the adaptor for a specific source and parameters.

        This method is typically called to prepare the adaptor for reading a particular
        `RasterSource` with specific `RasterLoadParams`. It returns a new instance
        of `ReaderDaskAdaptor` (or self) configured with these details.

        :param src: The `RasterSource` to be read.
        :param cfg: The `RasterLoadParams` for this read.
        :param ctx: The context for the reader driver.
        :param layer_name: Name for Dask layers.
        :param idx: Index for this source/task.
        :return: A `ReaderDaskAdaptor` instance configured for the read.
        """
        return ReaderDaskAdaptor(
            self._driver,
            self._env,
            ctx,
            src,
            cfg,
            layer_name=layer_name,
            idx=idx,
        )


def resolve_load_cfg(
    bands: dict[str, RasterBandMetadata],
    resampling: str | dict[str, str] | None = None,
    dtype: Band_DType | None = None,
    use_overviews: bool = True,
    nodata: float | None = None,
    fail_on_error: bool = True,
) -> dict[str, RasterLoadParams]:
    """
    Combine band metadata with user provided settings to produce load configuration.
    """

    def _dtype(name: str, band_dtype: str | None, fallback: str) -> str:
        if dtype is None:
            return with_default(band_dtype, fallback)
        if isinstance(dtype, dict):
            return str(
                with_default(
                    dtype.get(name, dtype.get("*", band_dtype)),
                    fallback,
                )
            )
        return str(dtype)

    def _resampling(name: str, fallback: str) -> str:
        if resampling is None:
            return fallback
        if isinstance(resampling, dict):
            return resampling.get(name, resampling.get("*", fallback))
        return resampling

    def _fill_value(band: RasterBandMetadata) -> float | None:
        if nodata is not None:
            return nodata
        return band.nodata

    def _resolve(name: str, band: RasterBandMetadata) -> RasterLoadParams:
        return RasterLoadParams(
            _dtype(name, band.data_type, "float32"),
            fill_value=_fill_value(band),
            use_overviews=use_overviews,
            resampling=_resampling(name, "nearest"),
            fail_on_error=fail_on_error,
            dims=band.dims,
        )

    return {name: _resolve(name, band) for name, band in bands.items()}


def resolve_src_nodata(
    nodata: Optional[float], cfg: RasterLoadParams
) -> Optional[float]:
    """
    Determine the definitive source nodata value based on configuration overrides.

    Priority:
    1. `cfg.src_nodata_override`
    2. `nodata` (passed in, usually from source metadata)
    3. `cfg.src_nodata_fallback`

    :param nodata: Nodata value from the source metadata, if available.
    :param cfg: RasterLoadParams containing potential overrides.
    :return: The resolved source nodata value, or None if not defined.
    """
    if cfg.src_nodata_override is not None:
        return cfg.src_nodata_override
    if nodata is not None:
        return nodata
    return cfg.src_nodata_fallback


def resolve_dst_dtype(src_dtype: str, cfg: RasterLoadParams) -> np.dtype:
    """
    Determine the destination data type for a band.

    If `cfg.dtype` is specified, it is used. Otherwise, `src_dtype` is used.

    :param src_dtype: The data type of the source band.
    :param cfg: RasterLoadParams which may specify a destination dtype.
    :return: The resolved numpy.dtype for the destination.
    """
    if cfg.dtype is None:
        return np.dtype(src_dtype)
    return np.dtype(cfg.dtype)


def resolve_dst_nodata(
    dst_dtype: np.dtype,
    cfg: RasterLoadParams,
    src_nodata: Optional[float] = None,
) -> Optional[float]:
    """
    Determine the nodata value to use for the destination array.

    Priority:
    1. `cfg.fill_value` (if specified in RasterLoadParams)
    2. `np.nan` if `dst_dtype` is float.
    3. `src_nodata` (if available and convertible to `dst_dtype`).
    4. None.

    :param dst_dtype: The numpy.dtype of the destination array.
    :param cfg: RasterLoadParams containing load settings.
    :param src_nodata: The nodata value from the source, if known.
    :return: The resolved destination nodata value, or None.
    """
    # 1. Configuration
    # 2. np.nan for float32 outputs
    # 3. Fall back to source nodata
    if cfg.fill_value is not None:
        return dst_dtype.type(cfg.fill_value)

    if dst_dtype.kind == "f":
        return np.nan

    if src_nodata is not None:
        return dst_dtype.type(src_nodata)

    return None


def resolve_dst_fill_value(
    dst_dtype: np.dtype,
    cfg: RasterLoadParams,
    src_nodata: Optional[float] = None,
) -> float:
    """
    Determine the fill value to use for initializing the destination array.

    This is typically the destination nodata value. If no nodata value is
    resolved for the destination, it defaults to 0.

    :param dst_dtype: The numpy.dtype of the destination array.
    :param cfg: RasterLoadParams containing load settings.
    :param src_nodata: The nodata value from the source, if known.
    :return: The value to use for filling the destination array.
    """
    nodata = resolve_dst_nodata(dst_dtype, cfg, src_nodata)
    if nodata is None:
        return dst_dtype.type(0)
    return nodata


def _selection_to_bands(selection: Any, n: int) -> list[int]:
    """
    Convert a band selection input into a list of 1-based band indices.

    :param selection: Band selection. Can be None (all bands), a list of bands,
                      an integer index, or a slice/array for numpy-style selection.
    :param n: Total number of bands available.
    :return: A list of 1-based band indices.
    """
    if selection is None:
        return list(range(1, n + 1))

    if isinstance(selection, list):
        return selection

    bidx = np.arange(1, n + 1)
    if isinstance(selection, int):
        return [int(bidx[selection])]
    return bidx[selection].tolist()


def resolve_band_query(
    src: RasterSource,
    n: int,
    selection: ReaderSubsetSelection | None = None,
) -> int | list[int]:
    """
    Resolve a band query for a given RasterSource.

    Determines which band(s) to load from a source that might be multi-band
    (e.g., a NetCDF file where `src.band == 0` indicates all bands or a selection)
    or single-band (where `src.band` is a 1-based index).

    :param src: The RasterSource.
    :param n: The number of available bands in the source if it's a multi-band file.
    :param selection: Optional selection criteria, used if `src.band == 0`.
    :return: A 1-based band index or a list of 1-based band indices.
    :raises ValueError: If `src.band` is out of range for the available bands.
    """
    if src.band > n:
        raise ValueError(
            f"Requested band {src.band} from {src.uri} with only {n} bands"
        )

    if src.band == 0:
        return _selection_to_bands(selection, n)

    meta = src.meta
    if meta is None:
        return src.band
    if meta.extra_dims:
        return [src.band]

    return src.band


def expand_selection(selection: Any, ydim: int) -> tuple[slice, ...]:
    """
    Add Y/X slices to selection tuple

    :param selection: Selection object
    :return: Tuple of slices
    """
    if selection is None:
        selection = ()
    if not isinstance(selection, tuple):
        selection = (selection,)

    prefix, postfix = selection[:ydim], selection[ydim:]
    return prefix + (slice(None), slice(None)) + postfix


def pick_overview(read_shrink: int, overviews: Sequence[int]) -> Optional[int]:
    """
    Select the best overview level based on the desired shrink factor.

    Finds the overview level in `overviews` that is closest to, but not
    larger than, `read_shrink`. Overviews are assumed to be sorted from
    smallest shrink factor to largest.

    :param read_shrink: The desired shrink factor (e.g., if asking for 1/8th resolution, shrink is 8).
    :param overviews: A sequence of available overview shrink factors (e.g., [2, 4, 8, 16]).
    :return: The index of the chosen overview in the `overviews` sequence, or None if no suitable overview.
    """
    if len(overviews) == 0 or read_shrink < overviews[0]:
        return None

    _idx = 0
    for idx, ovr in enumerate(overviews):
        if ovr > read_shrink:
            break
        _idx = idx

    return _idx


def same_nodata(a: Optional[float], b: Optional[float]) -> bool:
    """
    Compare two optional nodata values, correctly handling NaN.

    :param a: First nodata value.
    :param b: Second nodata value.
    :return: True if both are None, or if both are NaN, or if they are equal. False otherwise.
    """
    if a is None:
        return b is None
    if b is None:
        return False
    if math.isnan(a):
        return math.isnan(b)
    return a == b


def nodata_mask(pix: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """
    Create a boolean mask indicating nodata pixels in an array.

    Handles float arrays where nodata might be NaN, and integer arrays.
    If `nodata` is None, returns an all-False mask for float arrays if nodata is also NaN,
    otherwise for integer arrays it's an all-False mask.

    :param pix: The input numpy array.
    :param nodata: The nodata value to mask.
    :return: A boolean numpy array of the same shape as `pix`, True where pixels are nodata.
    """
    if pix.dtype.kind == "f":
        if nodata is None or math.isnan(nodata):
            return np.isnan(pix)
        return np.bitwise_or(np.isnan(pix), pix == nodata)
    if nodata is None:
        return np.zeros_like(pix, dtype="bool")
    return pix == nodata
