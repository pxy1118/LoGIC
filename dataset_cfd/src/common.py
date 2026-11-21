from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('CFD outputs must remain inside dataset_cfd')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def state(case, stage, **data):
    p = Path(case) / 'state.json'
    d = read(p) if p.exists() else {'created_utc': now(), 'stages': {}}
    d.update(stage=stage, updated_utc=now())
    d['stages'][stage] = {'utc': now(), **data}
    save(p, d)


@contextmanager
def lock(path):
    import msvcrt
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Lock outside CFD root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError('Another CFD writer holds this lock') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
