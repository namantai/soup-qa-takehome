"""Restore the selected PUBLIC DERIVATIVE artifacts; no downloads or execution."""
import hashlib
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'artifacts/public-repro-artifacts.zip'
DESTINATION = ROOT/'runs/returned/stage4-20261111T063626Z'
EXPECTED = '97f38aa5a9e09c102600f304de1bf871105b6aaede7609d1c03cf6cb1c239b5d'


def main():
    if DESTINATION.exists():
        raise FileExistsError('Returned artifacts already exist; do not overwrite them')
    actual = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise ValueError('Archive SHA-256 differs from the public derivative manifest')
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe archive member')
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        DESTINATION.mkdir(parents=True)
        archive.extractall(DESTINATION)
    print(f'Restored public derivative artifacts to {DESTINATION.relative_to(ROOT)}; SHA-256 {actual}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
