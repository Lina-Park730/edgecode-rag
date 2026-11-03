#!/usr/bin/env python3
"""Download the pinned Click reference source used by EdgeCode experiments."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://github.com/pallets/click/archive/refs/tags/8.1.7.zip'
SHA256 = '18f30d882d652f6b946613a045b7c55d270d85f3fbb218ab0d24c405cafa6a41'
NAME = 'click-8.1.7'


def install_archive(data: bytes, parent: Path) -> Path:
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Click archive SHA-256 mismatch; no files were installed.')
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / NAME
    if target.exists():
        raise FileExistsError(f'{target} already exists; it will not be overwritten.')
    with tempfile.TemporaryDirectory(dir=parent) as work:
        staging = Path(work)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or '..' in path.parts or not path.parts
                        or path.parts[0] != NAME or '\\' in item.filename
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError(f'Unsafe archive entry: {item.filename}')
                dest = staging.joinpath(*path.parts)
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, dest.open('wb') as output:
                        shutil.copyfileobj(source, output)
        extracted = staging / NAME
        for required in ['src/click/__init__.py', 'LICENSE.rst']:
            if not (extracted / required).is_file():
                raise ValueError(f'Archive is missing {required}')
        extracted.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP (same checksum required)')
    args = parser.parse_args()
    target = ROOT / 'repository' / NAME
    if target.exists():
        if (target/'src/click/__init__.py').is_file() and (target/'LICENSE.rst').is_file():
            print(f'Using existing source at {target}. No download or overwrite performed.')
            print('Existing file contents were not reverified; use a fresh directory for a verified download.')
            return 0
        parser.exit(1, f'Incomplete directory: {target}. Move it aside and rerun.\n')
    try:
        if args.archive:
            data = args.archive.read_bytes()
        else:
            print(f'Downloading Click 8.1.7 from {URL}')
            with urlopen(URL, timeout=120) as response:
                data = response.read()
        installed = install_archive(data, target.parent)
        print(f'Installed verified Click 8.1.7 source and license at {installed}')
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Download failed: {exc}\nYou can download the official ZIP manually and use --archive PATH.\n')


if __name__ == '__main__':
    raise SystemExit(main())
