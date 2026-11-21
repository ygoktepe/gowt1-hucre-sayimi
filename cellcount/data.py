"""Data access and reproducible frame selection for the CTC training archive."""

import csv
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import ZipFile


ARCHIVE_URL = 'https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip'
DEFAULT_SEED = 29033
FRAME_GAP = 5
BLOCK_LENGTH = 5
SELECTION_FIELDS = ('group', 'sequence', 'block', 'frame', 'filename', 'seed')


class Dataset:
    """Read individual TIFFs directly from a ZIP or an extracted dataset folder."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.archive = None

    def __enter__(self):
        if self.path.is_file():
            self.archive = ZipFile(self.path)
            self.root = 'Fluo-N2DH-GOWT1/'
        else:
            self.root = self.path
            if not (self.root / '01').is_dir():
                self.root /= 'Fluo-N2DH-GOWT1'
            if not (self.root / '01').is_dir():
                raise FileNotFoundError(f'CTC eğitim ZIP dosyası veya veri klasörü: {self.path}')
        return self

    def __exit__(self, *_):
        if self.archive is not None:
            self.archive.close()

    def frames(self, sequence):
        if self.archive is not None:
            folder = f'{self.root}{sequence}/'
            names = [name.removeprefix(folder) for name in self.archive.namelist()
                     if name.startswith(folder)]
        else:
            names = [path.name for path in (self.root / sequence).iterdir()]
        return {int(name[1:-4]): name for name in sorted(names)
                if re.fullmatch(r't\d+\.tif', name)}

    def _read_tiff(self, relative_path):
        from io import BytesIO
        import numpy as np
        from PIL import Image

        source = self.root / relative_path if self.archive is None else BytesIO(
            self.archive.read(self.root + relative_path)
        )
        with Image.open(source) as image:
            array = np.array(image)
        if array.ndim != 2 or array.dtype.kind != 'u':
            raise ValueError(f'Beklenen tek kanallı işaretsiz TIFF: {relative_path}')
        return array

    def image(self, row):
        return self._read_tiff(f"{row['sequence']}/{row['filename']}")

    def tracking_labels(self, row, *, allow_test=False):
        if row['sequence'] == '02' and not allow_test:
            raise PermissionError('Sekans 02 etiketi yalnız kilitten sonraki değerlendirmede okunabilir.')
        path = PurePosixPath(f"{row['sequence']}_GT", 'TRA', f"man_track{int(row['frame']):03d}.tif")
        return self._read_tiff(str(path))


def select_frames(data, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    first = data.frames('01')
    second = data.frames('02')
    first_ids = sorted(first)
    for _ in range(10_000):
        validation = sorted(rng.sample(first_ids, 5))
        eligible = [frame for frame in first_ids
                    if all(abs(frame - other) > FRAME_GAP for other in validation)]
        if len(eligible) >= 10:
            development = sorted(rng.sample(eligible, 10))
            break
    else:
        raise ValueError('Sekans 01 üzerinde aralıklı veri bölünmesi oluşturulamadı.')

    starts = [frame for frame in sorted(second)
              if all(frame + offset in second for offset in range(BLOCK_LENGTH))]
    for _ in range(10_000):
        blocks = sorted(rng.sample(starts, 3))
        if all(right - left >= FRAME_GAP + BLOCK_LENGTH
               for left, right in zip(blocks, blocks[1:])):
            break
    else:
        raise ValueError('Sekans 02 üzerinde aralıklı test blokları oluşturulamadı.')

    def row(group, sequence, frame, names, block=''):
        return dict(group=group, sequence=sequence, block=block, frame=frame,
                    filename=names[frame], seed=seed)

    return ([row('development', '01', frame, first) for frame in development]
            + [row('validation', '01', frame, first) for frame in validation]
            + [row('test', '02', start + offset, second, block)
               for block, start in enumerate(blocks, 1) for offset in range(BLOCK_LENGTH)])


def write_selection(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_selection(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return [dict(row, frame=int(row['frame']), seed=int(row['seed']),
                     block=int(row['block']) if row['block'] else '')
                for row in csv.DictReader(stream)]
