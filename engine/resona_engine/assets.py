"""Locate and install the shared feature-extractor assets (ContentVec, RMVPE).

Assets live in %LOCALAPPDATA%\\Resona\\assets. They can be imported from an existing
Vonovox install or downloaded from Hugging Face; either way every file is SHA-256 verified.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetFile:
    rel: str  # path relative to the assets root
    sha256: str
    url: str
    vonovox_rel: str  # path relative to a Vonovox `assets/` dir


ASSET_FILES: tuple[AssetFile, ...] = (
    AssetFile(
        "contentvec/config.json",
        "236411bad6d07b9ee685cf2dba66a0e3868fe5f7831ac485802985107c96585f",
        "https://huggingface.co/lengyue233/content-vec-best/resolve/main/config.json",
        "contentvec/config.json",
    ),
    AssetFile(
        "contentvec/pytorch_model.bin",
        "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e",
        "https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin",
        "contentvec/pytorch_model.bin",
    ),
    AssetFile(
        "rmvpe/rmvpe.pt",
        "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "rmvpe/rmvpe.pt",
    ),
)

# Known Vonovox install locations searched when no explicit path is given.
VONOVOX_CANDIDATES = (Path(r"C:\Pedro\AiVoiceChanger\Vonovox-1.6.9\Vonovox-1.6.9\assets"),)


@dataclass(frozen=True)
class AssetPaths:
    contentvec_dir: Path
    rmvpe: Path


class AssetError(RuntimeError):
    pass


def data_root() -> Path:
    base = os.environ.get("RESONA_DATA") or os.path.join(
        os.environ.get("LOCALAPPDATA", str(Path.home())), "Resona"
    )
    return Path(base)


def assets_root() -> Path:
    return data_root() / "assets"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing(root: Path | None = None) -> list[AssetFile]:
    root = root or assets_root()
    return [a for a in ASSET_FILES if not (root / a.rel).is_file()]


def resolve(root: Path | None = None) -> AssetPaths:
    """Return asset paths, importing from Vonovox if possible. Raises AssetError if absent."""
    root = root or assets_root()
    if missing(root):
        import_from_vonovox(root=root)
    if gone := missing(root):
        names = ", ".join(a.rel for a in gone)
        raise AssetError(f"missing assets: {names} (run `resona-engine assets fetch`)")
    return AssetPaths(contentvec_dir=root / "contentvec", rmvpe=root / "rmvpe" / "rmvpe.pt")


def _install(src: Path, asset: AssetFile, root: Path) -> None:
    digest = sha256_file(src)
    if digest != asset.sha256:
        raise AssetError(f"hash mismatch for {asset.rel}: {digest}")
    dst = root / asset.rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def import_from_vonovox(vonovox_assets: Path | None = None, root: Path | None = None) -> int:
    """Copy verified assets from a Vonovox `assets/` dir. Returns number of files installed."""
    root = root or assets_root()
    candidates = [vonovox_assets] if vonovox_assets else list(VONOVOX_CANDIDATES)
    installed = 0
    for cand in candidates:
        if not cand or not cand.is_dir():
            continue
        for asset in missing(root):
            src = cand / asset.vonovox_rel
            if src.is_file():
                _install(src, asset, root)
                installed += 1
    return installed


def fetch(root: Path | None = None, progress=None) -> None:
    """Download any missing assets from Hugging Face, verifying hashes."""
    root = root or assets_root()
    for asset in missing(root):
        dst = root / asset.rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".download")
        try:
            with urllib.request.urlopen(asset.url) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(asset.rel, done, total)
            _install(tmp, asset, root)
        finally:
            tmp.unlink(missing_ok=True)
