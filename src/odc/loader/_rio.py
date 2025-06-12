# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
odc.loader._rio
===============

This module provides the Rasterio-based reader driver (`RioDriver`) for `odc-loader`.
It handles opening raster files using Rasterio, managing GDAL environment settings,
and performing read operations, including reprojection and nodata handling.

Key components:
- `RioDriver`: Implements the `ReaderDriver` protocol using Rasterio.
- `RioReader`: The reader class instantiated by `RioDriver` for each source.
- `rio_read`: Core function performing the actual read operation with Rasterio.
- Environment management functions (`capture_rio_env`, `rio_env`, `configure_rio`,
  `configure_s3_access`) for setting up GDAL options and AWS credentials.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import rasterio
import rasterio.enums
import rasterio.env
import rasterio.warp
from odc.geo.converters import rio_geobox
from odc.geo.geobox import GeoBox
from odc.geo.overlap import ReprojectInfo, compute_reproject_roi
from odc.geo.roi import roi_is_empty, roi_shape, w_
from odc.geo.warp import resampling_s2rio
from rasterio.session import AWSSession, Session

from ._reader import (
    nodata_mask,
    pick_overview,
    resolve_band_query,
    resolve_dst_dtype,
    resolve_dst_nodata,
    resolve_src_nodata,
    same_nodata,
)
from .types import (
    DaskRasterReader,
    MDParser,
    RasterLoadParams,
    RasterSource,
    ReaderSubsetSelection,
)

log = logging.getLogger(__name__)

SECRET_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCESS_TOKEN",
    "AZURE_STORAGE_ACCESS_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_SAS",
    "GS_ACCESS_KEY_ID",
    "GS_SECRET_ACCESS_KEY",
    "OSS_ACCESS_KEY_ID",
    "OSS_SECRET_ACCESS_KEY",
    "SWIFT_AUTH_TOKEN",
)

SESSION_KEYS = (
    *SECRET_KEYS,
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_S3_ENDPOINT",
    "AWS_NO_SIGN_REQUEST",
    "AWS_REQUEST_PAYER",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_NO_SIGN_REQUEST",
    "OSS_ENDPOINT",
    "SWIFT_STORAGE_URL",
)
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
}

# pylint: disable=too-few-public-methods


class RioReader:
    """
    Reader part of RIO driver.
    """

    class LoaderState:
        """
        State shared across all RioReader instances for a single ``.load()`` operation.

        This can hold information like the target geobox and whether Dask is being used.
        Future extensions could include caching open file handles.
        """

        def __init__(self, geobox: GeoBox, is_dask: bool) -> None:
            """
            Initialize LoaderState.

            :param geobox: The target GeoBox for the load operation.
            :param is_dask: True if Dask is being used for this load.
            """
            self.geobox = geobox
            self.is_dask = is_dask

        def finalise(self) -> None:
            """Called when the load operation is complete."""
            pass

    def __init__(self, src: RasterSource, ctx: "RioReader.LoaderState") -> None:
        """
        Initialize RioReader.

        :param src: The `RasterSource` this reader will handle.
        :param ctx: The shared `LoaderState` for this load operation.
        """
        self._src = src
        self._ctx = ctx

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        dst: Optional[np.ndarray] = None,
        selection: Optional[ReaderSubsetSelection] = None,
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        """
        Read data from the `RasterSource` into the destination `GeoBox`.

        :param cfg: `RasterLoadParams` defining how to read the data.
        :param dst_geobox: The destination `GeoBox`.
        :param dst: Optional destination numpy array to read into.
        :param selection: Optional selection for subsetting.
        :return: Tuple of (destination_roi_slices, numpy_array_of_pixels).
        """
        return rio_read(self._src, cfg, dst_geobox, dst=dst, selection=selection)


class RioDriver:
    """
    Rasterio-based ReaderDriver implementation.

    This driver uses Rasterio to open and read various raster formats.
    It manages GDAL environment settings and AWS session configurations.
    """

    def new_load(
        self,
        geobox: GeoBox,
        *,
        chunks: None | Dict[str, int] = None,
    ) -> RioReader.LoaderState:
        """
        Initialize a new load operation.

        :param geobox: The target `GeoBox` for the entire load.
        :param chunks: Optional chunking information if Dask is used.
        :return: A `RioReader.LoaderState` instance for this load.
        """
        return RioReader.LoaderState(geobox, is_dask=chunks is not None)

    def finalise_load(self, load_state: RioReader.LoaderState) -> Any:
        """
        Finalize the load operation.

        :param load_state: The `RioReader.LoaderState` for this load.
        :return: Result of finalization (currently None).
        """
        return load_state.finalise()

    def capture_env(self) -> Dict[str, Any]:
        """
        Capture the current GDAL and AWS environment settings.

        :return: A dictionary of environment settings.
        """
        return capture_rio_env()

    @contextmanager
    def restore_env(
        self, env: Dict[str, Any], load_state: RioReader.LoaderState
    ) -> Iterator[RioReader.LoaderState]:
        """
        Context manager to temporarily restore GDAL/AWS environment for a read operation.

        :param env: Environment dictionary captured by `capture_env`.
        :param load_state: The current `RioReader.LoaderState`.
        :yields: The `load_state`.
        """
        with rio_env(**env):
            yield load_state

    def open(
        self,
        src: RasterSource,
        ctx: RioReader.LoaderState,
    ) -> RioReader:
        """
        Open a `RasterSource` for reading.

        :param src: The `RasterSource` to open.
        :param ctx: The current `RioReader.LoaderState`.
        :return: A `RioReader` instance.
        """
        return RioReader(src, ctx)

    @property
    def md_parser(self) -> MDParser | None:
        """The metadata parser associated with this driver (None for RioDriver)."""
        return None

    @property
    def dask_reader(self) -> DaskRasterReader | None:
        """The Dask-specific reader interface (None for basic RioDriver)."""
        return None


class _GlobalRioConfig:
    """Internal class to hold global Rasterio/GDAL configuration."""
    def __init__(self) -> None:
        self._configured = False
        self._aws: Optional[Dict[str, Any]] = None
        self._gdal_opts: Dict[str, Any] = {}

    def set(
        self,
        *,
        aws: Optional[Dict[str, Any]],
        gdal_opts: Dict[str, Any],
    ) -> None:
        """
        Set the global AWS and GDAL options.

        :param aws: AWS session parameters.
        :param gdal_opts: GDAL configuration options.
        """
        self._aws = {**aws} if aws is not None else None
        self._gdal_opts = {**gdal_opts}
        self._configured = True

    @property
    def configured(self) -> bool:
        """Check if global configuration has been set."""
        return self._configured

    def env(self) -> rasterio.env.Env:
        """
        Create a Rasterio Env based on the global configuration.

        Uses thread-local session management.
        :return: A `rasterio.env.Env` instance.
        """
        if self._configured is False:
            return rasterio.env.Env(_local.session())

        session: Optional[Session] = None
        if self._aws is not None:
            session = AWSSession(**self._aws)
        return rasterio.env.Env(_local.session(session), **self._gdal_opts)


_CFG = _GlobalRioConfig()


class ThreadSession(threading.local):
    """
    Caches Session between rio_env calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[Session] = None
        self._aws: Optional[Dict[str, Any]] = None

    @property
    def configured(self) -> bool:
        return self._session is not None

    def reset(self) -> None:
        """Reset the thread-local session and GDAL environment."""
        self._session = None
        self._aws = None
        if rasterio.env.hasenv():
            rasterio.env.delenv()

    def session(self, session: Union[Dict[str, Any], Session] = None) -> Session:
        """
        Get or initialize the Rasterio session for the current thread.

        If a session is provided, it may be used or cached.
        Manages the global GDAL environment activation (`rasterio.env.defenv()`).

        :param session: Optional session info (dict for AWS, or `rasterio.Session` instance).
        :return: The active `rasterio.Session`.
        """
        if self._session is None:
            # first call in this thread
            # 1. Start GDAL environment
            rasterio.env.defenv()

            if session is None:
                # Figure out session from environment variables
                with rasterio.env.Env() as env:
                    self._session = env.session
            else:
                if isinstance(session, dict):
                    self._aws = session
                    session = AWSSession(**session)
                self._session = session

            assert self._session is not None
            return self._session

        if session is not None:
            if isinstance(session, Session):
                return session
            # TODO: cache more than one session?
            if session == self._aws:
                return self._session
            return AWSSession(**session)

        return self._session


_local = ThreadSession()


def _sanitize(opts: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Sanitize a dictionary by replacing values of specified keys.

    :param opts: Dictionary to sanitize.
    :param keys: Keys whose values should be replaced with "xx..xx".
    :return: Sanitized dictionary.
    """
    return {k: (v if k not in keys else "xx..xx") for k, v in opts.items()}


def get_rio_env(sanitize: bool = True, no_session_keys: bool = False) -> Dict[str, Any]:
    """
    Get GDAL params configured by rasterio for the current thread.

    :param sanitize: If True replace sensitive Values with 'x'
    :param no_session_keys: Remove keys that need to be supplied via Session classes.
    """

    if not rasterio.env.hasenv():
        return {}

    opts = rasterio.env.getenv()
    if no_session_keys:
        opts = {k: v for k, v in opts.items() if k not in SESSION_KEYS}
    if sanitize:
        opts = _sanitize(opts, SECRET_KEYS)

    return opts


def rio_env(session=None, **kw):
    """
    Wraps rasterio.env.Env.

    re-uses GDAL environment and session between calls.
    """
    if session is None:
        session = kw.pop("_aws", None)
    return rasterio.env.Env(_local.session(session), **kw)


def _set_default_rio_config(
    aws: Optional[Dict[str, Any]] = None,
    cloud_defaults: bool = False,
    **kwargs,
) -> None:
    """
    Internal helper to set the global Rasterio configuration (_CFG).

    :param aws: AWS session parameters.
    :param cloud_defaults: Whether to apply GDAL cloud default settings.
    :param kwargs: Other GDAL options.
    """
    opts = {**GDAL_CLOUD_DEFAULTS, **kwargs} if cloud_defaults else {**kwargs}
    _CFG.set(aws=aws, gdal_opts=opts)


def configure_rio(
    *,
    cloud_defaults: bool = False,
    verbose: bool = False,
    aws: Optional[Dict[str, Any]] = None,
    **params,
) -> None:
    """
    Change GDAL/rasterio configuration.

    Change can be applied locally or on a Dask cluster using ``WorkerPlugin`` mechanism.

    :param cloud_defaults: When ``True`` enable common cloud settings in GDAL
    :param verbose: Dump GDAL environment settings to stdout
    :param aws: Arguments for :py:class:`rasterio.session.AWSSession`
    """
    # backward compatible flags that don't make a difference anymore
    activate = params.pop("activate", None)
    client = params.pop("client", None)
    if client is not None:
        pass
    if activate is not None:
        pass

    _set_default_rio_config(cloud_defaults=cloud_defaults, aws=aws, **params)
    if verbose:
        with _CFG.env():
            _dump_rio_config()


def _dump_rio_config() -> None:
    """Dump current Rasterio/GDAL environment configuration to stdout."""
    cfg = get_rio_env()
    if not cfg:
        return
    nw = max(len(k) for k in cfg)
    for k, v in cfg.items():
        print(f"{k:<{nw}} = {v}")


def configure_s3_access(
    profile: Optional[str] = None,
    region_name: str = "auto",
    aws_unsigned: Optional[bool] = None,
    requester_pays: bool = False,
    cloud_defaults: bool = True,
    **gdal_opts,
):
    """
    Credentialize for S3 bucket access or configure public access.

    This function obtains credentials for S3 access and passes them on to
    processing threads, either local or on dask cluster.


    .. note::

       if credentials are STS based they will eventually expire, currently
       this case is not handled very well, reads will just start failing
       eventually and will never recover.

    :param profile:        AWS profile name to use
    :param region_name:    Default region_name to use if not configured for a given/default AWS profile
    :param aws_unsigned:   If ``True`` don't bother with credentials when reading from S3
    :param requester_pays: Needed when accessing requester pays buckets

    :param cloud_defaults: Assume files are in the cloud native format, i.e. no side-car files, disables
                           looking for side-car files, makes things faster but won't work for files
                           that do have side-car files with extra metadata.

    :param gdal_opts:      Any other option to pass to GDAL environment setup

    :returns: credentials object or ``None`` if ``aws_unsigned=True``
    """
    # pylint: disable=import-outside-toplevel
    try:
        from ._aws import get_aws_settings
    except ImportError as e:
        raise ImportError(
            "botocore is required to configure s3 access. "
            "Install botocore directly or via `pip install 'odc-stac[botocore]'"
        ) from e

    aws, creds = get_aws_settings(
        profile=profile,
        region_name=region_name,
        aws_unsigned=aws_unsigned,
        requester_pays=requester_pays,
    )

    _set_default_rio_config(aws=aws, cloud_defaults=cloud_defaults, **gdal_opts)
    return creds


def _reproject_info_from_rio(
    rdr: rasterio.DatasetReader, dst_geobox: GeoBox, ttol: float
) -> ReprojectInfo:
    """
    Compute reprojection information from a Rasterio reader to a destination GeoBox.

    :param rdr: Open `rasterio.DatasetReader`.
    :param dst_geobox: Target `GeoBox`.
    :param ttol: Tolerance for considering transforms as identical.
    :return: `ReprojectInfo` detailing the operation.
    """
    return compute_reproject_roi(rio_geobox(rdr), dst_geobox, ttol=ttol)


def _do_read(
    src: rasterio.Band,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    rr: ReprojectInfo,
    dst: Optional[np.ndarray] = None,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """
    Perform the actual Rasterio read/reproject operation for a single band or band selection.

    Handles reading data, remapping nodata values if necessary, and reprojecting.

    :param src: `rasterio.Band` object (can represent multiple bands).
    :param cfg: `RasterLoadParams` for the operation.
    :param dst_geobox: Target `GeoBox`.
    :param rr: `ReprojectInfo` detailing the reprojection.
    :param dst: Optional destination numpy array.
    :return: Tuple of (destination_roi_slices, numpy_array_of_pixels).
    """
    resampling = resampling_s2rio(cfg.resampling)
    rdr = src.ds
    roi_dst: tuple[slice, slice] = rr.roi_dst  # type: ignore
    ndim = 2
    prefix: tuple[int, ...] = ()

    if isinstance(src.bidx, int):
        src_nodata0 = rdr.nodatavals[src.bidx - 1]
    else:
        ndim = 3
        prefix = (len(src.bidx),)
        (src_nodata0,) = (rdr.nodatavals[b - 1] for b in src.bidx[:1])

    if dst is not None:
        # Assumes Y,X or B,Y,X order
        assert dst.ndim == ndim
        _dst = dst[(...,) + roi_dst]
    else:
        shape = prefix + roi_shape(rr.roi_dst)
        _dst = np.ndarray(shape, dtype=resolve_dst_dtype(src.dtype, cfg))

    src_nodata = resolve_src_nodata(src_nodata0, cfg)
    dst_nodata = resolve_dst_nodata(_dst.dtype, cfg, src_nodata)

    if roi_is_empty(roi_dst):
        return (roi_dst, _dst)

    if roi_is_empty(rr.roi_src):
        # no overlap case
        if dst_nodata is not None:
            np.copyto(_dst, dst_nodata)
        return (roi_dst, _dst)

    if rr.paste_ok and rr.read_shrink == 1:
        rdr.read(src.bidx, out=_dst, window=w_[rr.roi_src])

        if dst_nodata is not None and not same_nodata(src_nodata, dst_nodata):
            # remap nodata from source to output
            np.copyto(_dst, dst_nodata, where=nodata_mask(_dst, src_nodata))
    else:
        # some form of reproject
        # TODO: support read with integer shrink then reproject more
        # TODO: deal with int8 inputs

        rasterio.warp.reproject(
            src,
            _dst,
            src_nodata=src_nodata,
            dst_crs=str(dst_geobox.crs),
            dst_transform=dst_geobox[rr.roi_dst].transform,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return (roi_dst, _dst)


def rio_read(
    src: RasterSource,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    dst: Optional[np.ndarray] = None,
    selection: Optional[ReaderSubsetSelection] = None,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """
    Internal read method.

    Returns part of the destination image that overlaps with the given source.

    .. code-block: python

       cfg, geobox, sources = ...
       mosaic = np.full(geobox.shape, cfg.fill_value, dtype=cfg.dtype)

       for src in sources:
           roi, pix = rio_read(src, cfg, geobox)
           assert mosaic[roi].shape == pix.shape
           assert pix.dtype == mosaic.dtype

           # paste over destination pixels that are empty
           np.copyto(mosaic[roi], pix, where=(mosaic[roi] == nodata))
           # OR
           mosaic[roi] = pix  # if sources are true tiles (no overlaps)

    """
    ydim = src.ydim

    def prep_dst(dst: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if dst is None:
            return None
        if dst.ndim == 2 or ydim == 1:
            # Y,X or B,Y,X
            return dst

        # Supplied as Y,X,B, but we need B,Y,X
        assert ydim == 0 and dst.ndim == 3
        return dst.transpose([2, 0, 1])

    def fixup_out(
        x: tuple[tuple[slice, slice], np.ndarray],
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        roi, out = x
        if out.ndim == 2 or ydim == 1:
            # Y,X or B,Y,X
            return roi, out

        # must be Y,X,B on output
        if dst is not None:
            return roi, dst[roi]

        assert ydim == 0 and out.ndim == 3
        # B,Y,X -> Y,X,B
        return roi, out.transpose([1, 2, 0])

    try:
        return fixup_out(
            _rio_read(src, cfg, dst_geobox, prep_dst(dst), selection=selection)
        )
    except (
        rasterio.errors.RasterioIOError,
        rasterio.errors.RasterBlockError,
        rasterio.errors.WarpOperationError,
        rasterio.errors.WindowEvaluationError,
    ) as e:
        if cfg.fail_on_error:
            log.error(
                "Aborting load due to failure while reading: %s:%d",
                src.uri,
                src.band,
            )
            raise e
    except rasterio.errors.RasterioError as e:
        if cfg.fail_on_error:
            log.error(
                "Aborting load due to some rasterio error: %s:%d",
                src.uri,
                src.band,
            )
            raise e

    # Failed to read, but asked to continue
    log.warning("Ignoring read failure while reading: %s:%d", src.uri, src.band)

    # TODO: capture errors somehow

    if dst is not None:
        out = dst[0:0, 0:0]
    else:
        out = np.ndarray((0, 0), dtype=cfg.dtype)

    return np.s_[0:0, 0:0], out


def _rio_read(
    src: RasterSource,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    dst: Optional[np.ndarray] = None,
    selection: Optional[ReaderSubsetSelection] = None,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """
    Helper for `rio_read` that handles opening the dataset and overview selection.

    Manages opening the rasterio dataset, selecting an overview if appropriate,
    and then calls `_do_read` to perform the read operation.

    :param src: `RasterSource` to read from.
    :param cfg: `RasterLoadParams` for the read.
    :param dst_geobox: Target `GeoBox`.
    :param dst: Optional destination numpy array.
    :param selection: Optional band/subset selection.
    :return: Tuple of (destination_roi_slices, numpy_array_of_pixels).
    """
    # if resampling is `nearest` then ignore sub-pixel translation when deciding
    # whether we can just paste source into destination
    ttol = 0.9 if cfg.nearest else 0.05

    with rasterio.open(src.uri, "r", sharing=False) as rdr:
        assert isinstance(rdr, rasterio.DatasetReader)
        ovr_idx: Optional[int] = None

        bidx = resolve_band_query(src, rdr.count, selection=selection)
        rr = _reproject_info_from_rio(rdr, dst_geobox, ttol=ttol)

        if cfg.use_overviews and rr.read_shrink > 1:
            first_band = bidx if isinstance(bidx, int) else bidx[0]
            ovr_idx = pick_overview(rr.read_shrink, rdr.overviews(first_band))

        if ovr_idx is None:
            with rio_env(VSI_CACHE=False):
                return _do_read(rasterio.band(rdr, bidx), cfg, dst_geobox, rr, dst=dst)

        # read from overview
        with rasterio.open(
            src.uri, "r", sharing=False, overview_level=ovr_idx
        ) as rdr_ovr:
            rr = _reproject_info_from_rio(rdr, dst_geobox, ttol=ttol)
            with rio_env(VSI_CACHE=False):
                return _do_read(
                    rasterio.band(rdr_ovr, bidx), cfg, dst_geobox, rr, dst=dst
                )


def capture_rio_env() -> Dict[str, Any]:
    """
    Capture GDAL environment settings relevant for Rasterio operations.

    This includes global GDAL options set via `configure_rio` and
    any options currently active in the thread-local Rasterio environment.
    Excludes GDAL_DATA to avoid issues if workers are on different machines.

    :return: A dictionary of GDAL environment options.
    """
    # pylint: disable=protected-access
    if _CFG._configured:
        env = {**_CFG._gdal_opts, "_aws": _CFG._aws}
    else:
        env = {}

    env.update(get_rio_env(sanitize=False, no_session_keys=True))
    # don't want that copied across to workers who might be on different machine
    env.pop("GDAL_DATA", None)

    if len(env) == 0:
        # not customized, supply defaults
        return {**GDAL_CLOUD_DEFAULTS}

    return env
