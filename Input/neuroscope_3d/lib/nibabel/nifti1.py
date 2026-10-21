"""
Standard NIfTI-1 reader and writer implementation for nibabel compatibility.
Supports .nii and .nii.gz files, endianness detection, sform/qform affines,
data proxies, and error handling.
"""

import gzip
import io
import math
import os
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .filebasedimages import ImageFileError
from .spatialimages import HeaderDataError

NIFTI1_STRUCT_FMT = (
    "i10s18sihcc8h3fhhhh8ffffhbbf fffii 80s24shhffffff4f4f4f16s4s".replace(" ", "")
)
HEADER_SIZE = 348

DT_MAP = {
    2: np.dtype("uint8"),
    4: np.dtype("int16"),
    8: np.dtype("int32"),
    16: np.dtype("float32"),
    64: np.dtype("float64"),
    256: np.dtype("int8"),
    512: np.dtype("uint16"),
    768: np.dtype("uint32"),
    1024: np.dtype("int64"),
    1280: np.dtype("uint64"),
}

INV_DT_MAP = {
    np.dtype("uint8"): (2, 8),
    np.dtype("int16"): (4, 16),
    np.dtype("int32"): (8, 32),
    np.dtype("float32"): (16, 32),
    np.dtype("float64"): (64, 64),
    np.dtype("int8"): (256, 8),
    np.dtype("uint16"): (512, 16),
    np.dtype("uint32"): (768, 32),
    np.dtype("int64"): (1024, 64),
    np.dtype("uint64"): (1280, 64),
}


class Nifti1Header:
    """Represents a NIfTI-1 header."""

    def __init__(self, endian="<"):
        self.endian = endian
        self.fields = {}
        self._init_defaults()

    def _init_defaults(self):
        self.fields = {
            "sizeof_hdr": 348,
            "data_type": b"",
            "db_name": b"",
            "extents": 0,
            "session_error": 0,
            "regular": b"r",
            "dim_info": b"\x00",
            "dim": [3, 1, 1, 1, 1, 1, 1, 1],
            "intent_p1": 0.0,
            "intent_p2": 0.0,
            "intent_p3": 0.0,
            "intent_code": 0,
            "datatype": 16,
            "bitpix": 32,
            "slice_start": 0,
            "pixdim": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "vox_offset": 352.0,
            "scl_slope": 1.0,
            "scl_inter": 0.0,
            "slice_end": 0,
            "slice_code": 0,
            "xyzt_units": 2,  # mm
            "cal_max": 0.0,
            "cal_min": 0.0,
            "slice_duration": 0.0,
            "toffset": 0.0,
            "glmax": 0,
            "glmin": 0,
            "descrip": b"",
            "aux_file": b"",
            "qform_code": 1,
            "sform_code": 1,
            "quatern_b": 0.0,
            "quatern_c": 0.0,
            "quatern_d": 0.0,
            "qoffset_x": 0.0,
            "qoffset_y": 0.0,
            "qoffset_z": 0.0,
            "srow_x": [1.0, 0.0, 0.0, 0.0],
            "srow_y": [0.0, 1.0, 0.0, 0.0],
            "srow_z": [0.0, 0.0, 1.0, 0.0],
            "intent_name": b"",
            "magic": b"n+1\x00",
        }

    def __getitem__(self, item):
        return self.fields[item]

    def __setitem__(self, key, value):
        self.fields[key] = value

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "Nifti1Header":
        if len(raw_bytes) < HEADER_SIZE:
            raise HeaderDataError(
                f"Header size too small: {len(raw_bytes)} bytes (expected {HEADER_SIZE})"
            )

        size_le = struct.unpack("<i", raw_bytes[:4])[0]
        size_be = struct.unpack(">i", raw_bytes[:4])[0]

        if size_le == 348:
            endian = "<"
        elif size_be == 348:
            endian = ">"
        else:
            raise HeaderDataError(
                f"Invalid NIfTI-1 sizeof_hdr: le={size_le}, be={size_be}. Expected 348."
            )

        hdr = cls(endian=endian)
        try:
            unpacked = struct.unpack(endian + NIFTI1_STRUCT_FMT, raw_bytes[:348])
        except struct.error as e:
            raise HeaderDataError(f"Corrupt NIfTI-1 header struct: {e}")

        hdr.fields["sizeof_hdr"] = unpacked[0]
        hdr.fields["data_type"] = unpacked[1]
        hdr.fields["db_name"] = unpacked[2]
        hdr.fields["extents"] = unpacked[3]
        hdr.fields["session_error"] = unpacked[4]
        hdr.fields["regular"] = unpacked[5]
        hdr.fields["dim_info"] = unpacked[6]
        hdr.fields["dim"] = list(unpacked[7:15])
        hdr.fields["intent_p1"] = unpacked[15]
        hdr.fields["intent_p2"] = unpacked[16]
        hdr.fields["intent_p3"] = unpacked[17]
        hdr.fields["intent_code"] = unpacked[18]
        hdr.fields["datatype"] = unpacked[19]
        hdr.fields["bitpix"] = unpacked[20]
        hdr.fields["slice_start"] = unpacked[21]
        hdr.fields["pixdim"] = list(unpacked[22:30])
        hdr.fields["vox_offset"] = unpacked[30]
        hdr.fields["scl_slope"] = unpacked[31]
        hdr.fields["scl_inter"] = unpacked[32]
        hdr.fields["slice_end"] = unpacked[33]
        hdr.fields["slice_code"] = unpacked[34]
        hdr.fields["xyzt_units"] = unpacked[35]
        hdr.fields["cal_max"] = unpacked[36]
        hdr.fields["cal_min"] = unpacked[37]
        hdr.fields["slice_duration"] = unpacked[38]
        hdr.fields["toffset"] = unpacked[39]
        hdr.fields["glmax"] = unpacked[40]
        hdr.fields["glmin"] = unpacked[41]
        hdr.fields["descrip"] = unpacked[42]
        hdr.fields["aux_file"] = unpacked[43]
        hdr.fields["qform_code"] = unpacked[44]
        hdr.fields["sform_code"] = unpacked[45]
        hdr.fields["quatern_b"] = unpacked[46]
        hdr.fields["quatern_c"] = unpacked[47]
        hdr.fields["quatern_d"] = unpacked[48]
        hdr.fields["qoffset_x"] = unpacked[49]
        hdr.fields["qoffset_y"] = unpacked[50]
        hdr.fields["qoffset_z"] = unpacked[51]
        hdr.fields["srow_x"] = list(unpacked[52:56])
        hdr.fields["srow_y"] = list(unpacked[56:60])
        hdr.fields["srow_z"] = list(unpacked[60:64])
        hdr.fields["intent_name"] = unpacked[64]
        hdr.fields["magic"] = unpacked[65]

        magic = hdr.fields["magic"].rstrip(b"\x00")
        if magic not in (b"n+1", b"ni1"):
            raise HeaderDataError(f"Invalid NIfTI magic: {hdr.fields['magic']!r}")

        return hdr

    def to_bytes(self) -> bytes:
        endian = self.endian
        vals = [
            self.fields["sizeof_hdr"],
            self.fields["data_type"][:10].ljust(10, b"\x00"),
            self.fields["db_name"][:18].ljust(18, b"\x00"),
            self.fields["extents"],
            self.fields["session_error"],
            self.fields["regular"][:1],
            self.fields["dim_info"][:1],
            *self.fields["dim"][:8],
            self.fields["intent_p1"],
            self.fields["intent_p2"],
            self.fields["intent_p3"],
            self.fields["intent_code"],
            self.fields["datatype"],
            self.fields["bitpix"],
            self.fields["slice_start"],
            *self.fields["pixdim"][:8],
            self.fields["vox_offset"],
            self.fields["scl_slope"],
            self.fields["scl_inter"],
            self.fields["slice_end"],
            self.fields["slice_code"],
            self.fields["xyzt_units"],
            self.fields["cal_max"],
            self.fields["cal_min"],
            self.fields["slice_duration"],
            self.fields["toffset"],
            self.fields["glmax"],
            self.fields["glmin"],
            self.fields["descrip"][:80].ljust(80, b"\x00"),
            self.fields["aux_file"][:24].ljust(24, b"\x00"),
            self.fields["qform_code"],
            self.fields["sform_code"],
            self.fields["quatern_b"],
            self.fields["quatern_c"],
            self.fields["quatern_d"],
            self.fields["qoffset_x"],
            self.fields["qoffset_y"],
            self.fields["qoffset_z"],
            *self.fields["srow_x"][:4],
            *self.fields["srow_y"][:4],
            *self.fields["srow_z"][:4],
            self.fields["intent_name"][:16].ljust(16, b"\x00"),
            self.fields["magic"][:4].ljust(4, b"\x00"),
        ]
        return struct.pack(endian + NIFTI1_STRUCT_FMT, *vals)

    def get_data_shape(self) -> Tuple[int, ...]:
        ndim = self.fields["dim"][0]
        if ndim < 1:
            return ()
        return tuple(self.fields["dim"][1 : ndim + 1])

    def get_zooms(self) -> Tuple[float, ...]:
        ndim = self.fields["dim"][0]
        if ndim < 1:
            return ()
        return tuple(self.fields["pixdim"][1 : ndim + 1])

    def get_sform(self) -> Optional[np.ndarray]:
        if self.fields["sform_code"] > 0:
            return np.array(
                [
                    self.fields["srow_x"],
                    self.fields["srow_y"],
                    self.fields["srow_z"],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        return None

    def get_qform(self) -> Optional[np.ndarray]:
        if self.fields["qform_code"] > 0:
            b = self.fields["quatern_b"]
            c = self.fields["quatern_c"]
            d = self.fields["quatern_d"]
            val = 1.0 - (b * b + c * c + d * d)
            a = math.sqrt(val) if val > 0.0 else 0.0

            r11 = a * a + b * b - c * c - d * d
            r12 = 2.0 * b * c - 2.0 * a * d
            r13 = 2.0 * b * d + 2.0 * a * c

            r21 = 2.0 * b * c + 2.0 * a * d
            r22 = a * a + c * c - b * b - d * d
            r23 = 2.0 * c * d - 2.0 * a * b

            r31 = 2.0 * b * d - 2.0 * a * c
            r32 = 2.0 * c * d + 2.0 * a * b
            r33 = a * a + d * d - c * c - b * b

            dx = self.fields["pixdim"][1]
            dy = self.fields["pixdim"][2]
            qfac = -1.0 if self.fields["pixdim"][0] < 0 else 1.0
            dz = self.fields["pixdim"][3] * qfac

            return np.array(
                [
                    [r11 * dx, r12 * dy, r13 * dz, self.fields["qoffset_x"]],
                    [r21 * dx, r22 * dy, r23 * dz, self.fields["qoffset_y"]],
                    [r31 * dx, r32 * dy, r33 * dz, self.fields["qoffset_z"]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        return None

    def get_base_affine(self) -> np.ndarray:
        sform = self.get_sform()
        if sform is not None:
            return sform
        qform = self.get_qform()
        if qform is not None:
            return qform
        zooms = self.get_zooms()
        diag = [zooms[i] if i < len(zooms) else 1.0 for i in range(3)] + [1.0]
        return np.diag(diag)

    def get_best_affine(self) -> np.ndarray:
        return self.get_base_affine()

    def set_sform(self, affine: np.ndarray, code: int = 1):
        self.fields["sform_code"] = code
        self.fields["srow_x"] = [float(x) for x in affine[0, :4]]
        self.fields["srow_y"] = [float(x) for x in affine[1, :4]]
        self.fields["srow_z"] = [float(x) for x in affine[2, :4]]

    def set_qform(self, affine: np.ndarray, code: int = 1):
        self.fields["qform_code"] = code
        self.fields["qoffset_x"] = float(affine[0, 3])
        self.fields["qoffset_y"] = float(affine[1, 3])
        self.fields["qoffset_z"] = float(affine[2, 3])
        dx = float(np.linalg.norm(affine[:3, 0]))
        dy = float(np.linalg.norm(affine[:3, 1]))
        dz = float(np.linalg.norm(affine[:3, 2]))
        if dx == 0:
            dx = 1.0
        if dy == 0:
            dy = 1.0
        if dz == 0:
            dz = 1.0
        self.fields["pixdim"][1] = dx
        self.fields["pixdim"][2] = dy
        self.fields["pixdim"][3] = dz


class _ArrayProxy:
    """Proxy object allowing lazy or sliced loading of voxel data."""

    def __init__(self, filename: Union[str, Path], header: Nifti1Header):
        self.filename = str(filename)
        self.header = header
        self.shape = header.get_data_shape()
        self.ndim = len(self.shape)
        dt_code = header.fields["datatype"]
        if dt_code not in DT_MAP:
            raise HeaderDataError(f"Unsupported NIfTI datatype code: {dt_code}")
        self.dtype = DT_MAP[dt_code].newbyteorder(header.endian)
        self.vox_offset = int(header.fields["vox_offset"])

    def _open_stream(self):
        try:
            if self.filename.endswith(".gz"):
                return gzip.GzipFile(self.filename, "rb")
            else:
                return open(self.filename, "rb")
        except gzip.BadGzipFile as e:
            raise ImageFileError(f"Corrupt gzip archive: {e}")
        except Exception as e:
            raise ImageFileError(f"Cannot open file {self.filename}: {e}")

    def __array__(self, dtype=None):
        data = self.read_all()
        if dtype is not None:
            return data.astype(dtype, copy=False)
        return data

    def read_all(self) -> np.ndarray:
        with self._open_stream() as f:
            try:
                f.seek(self.vox_offset)
            except Exception as e:
                raise ImageFileError(f"Failed seeking to voxel offset {self.vox_offset}: {e}")

            expected_voxels = 1
            for d in self.shape:
                expected_voxels *= d
            expected_bytes = expected_voxels * self.dtype.itemsize

            raw = f.read(expected_bytes)
            if len(raw) < expected_bytes:
                raise ImageFileError(
                    f"Truncated voxel data: expected {expected_bytes} bytes, received {len(raw)} bytes"
                )

            slope = self.header.fields["scl_slope"]
            inter = self.header.fields["scl_inter"]
            arr = np.frombuffer(raw, dtype=self.dtype).reshape(self.shape, order="F")

            if slope != 0.0 and (slope != 1.0 or inter != 0.0):
                arr = arr.astype(np.float64) * slope + inter

            return arr


class Nifti1Image:
    """Represents a loaded or new NIfTI-1 image."""

    def __init__(
        self,
        dataobj: Union[np.ndarray, _ArrayProxy],
        affine: np.ndarray,
        header: Optional[Nifti1Header] = None,
    ):
        self.dataobj = dataobj
        self.affine = np.asarray(affine, dtype=np.float64)
        if header is not None:
            self.header = header
        else:
            self.header = Nifti1Header()
            self._sync_header_with_data()

    def _sync_header_with_data(self):
        shape = self.dataobj.shape
        ndim = len(shape)
        self.header.fields["dim"] = [ndim] + list(shape) + [1] * (7 - ndim)

        dt = getattr(self.dataobj, "dtype", np.dtype("float32"))
        if dt in INV_DT_MAP:
            code, bitpix = INV_DT_MAP[dt]
        elif np.issubdtype(dt, np.floating):
            code, bitpix = 16, 32
        elif np.issubdtype(dt, np.integer):
            code, bitpix = 4, 16
        else:
            code, bitpix = 16, 32

        self.header.fields["datatype"] = code
        self.header.fields["bitpix"] = bitpix
        self.header.set_sform(self.affine, code=1)
        self.header.set_qform(self.affine, code=1)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.dataobj.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_fdata(self, dtype=np.float64) -> np.ndarray:
        if isinstance(self.dataobj, _ArrayProxy):
            arr = self.dataobj.read_all()
        else:
            arr = np.asarray(self.dataobj)
        return arr.astype(dtype, copy=False)

    def to_filename(self, filename: Union[str, Path]):
        save(self, filename)


def load(filename: Union[str, Path]) -> Nifti1Image:
    """Load a NIfTI-1 image from file."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ImageFileError(f"Unsupported file extension: {name}")

    try:
        if name.endswith(".gz"):
            try:
                with gzip.GzipFile(str(path), "rb") as gz:
                    raw_header = gz.read(HEADER_SIZE)
            except gzip.BadGzipFile as e:
                raise ImageFileError(f"Corrupt gzip archive: {e}")
            except EOFError as e:
                raise ImageFileError(f"Truncated gzip file: {e}")
        else:
            with open(str(path), "rb") as f:
                raw_header = f.read(HEADER_SIZE)
    except ImageFileError:
        raise
    except Exception as e:
        raise ImageFileError(f"Failed opening file: {e}")

    if len(raw_header) < HEADER_SIZE:
        raise ImageFileError(
            f"File too small for NIfTI header: got {len(raw_header)} bytes, expected at least {HEADER_SIZE}"
        )

    try:
        header = Nifti1Header.from_bytes(raw_header)
    except HeaderDataError as e:
        raise ImageFileError(f"Invalid NIfTI header: {e}")
    except Exception as e:
        raise ImageFileError(f"Corrupt NIfTI header: {e}")

    affine = header.get_base_affine()
    proxy = _ArrayProxy(str(path), header)
    return Nifti1Image(proxy, affine, header)


def save(img: Nifti1Image, filename: Union[str, Path]):
    """Save a Nifti1Image to .nii or .nii.gz."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = img.header
    header.fields["vox_offset"] = 352.0
    header.set_sform(img.affine, code=1)
    header.set_qform(img.affine, code=1)

    raw_header = header.to_bytes()
    extension = b"\x00\x00\x00\x00"

    data = img.get_fdata()
    dt_code = header.fields["datatype"]
    dtype = DT_MAP.get(dt_code, np.dtype("float32"))
    raw_data = np.asarray(data, dtype=dtype, order="F").tobytes()

    payload = raw_header + extension + raw_data

    if str(filename).lower().endswith(".gz"):
        with gzip.GzipFile(str(filename), "wb") as gz:
            gz.write(payload)
    else:
        with open(str(filename), "wb") as f:
            f.write(payload)
