"""Explicitly fetch the pinned public research kernel set; never run implicitly.

Default data/kernels is ignored by Git. Existing different files are preserved.
The Ceres kernel is the user's archived 2025-2045 snapshot, not a newly computed
orbit. Exact public URLs, expected byte counts and hashes live in the manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def fetch(directory, mission=False):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'configs/kernel-sources.json').read_text())
    if mission:
        manifest['kernels'] += json.loads((ROOT/'examples/missions/kernel-sources.json').read_text())['kernels']
    for item in manifest['kernels']:
        path=directory/item['filename']
        if path.exists():
            if digest(path)!=item['sha256']:
                raise ValueError(f'Existing kernel differs: {path}. Preserve it and choose an empty directory.')
            print('Verified',path,flush=True);continue
        print('Downloading',item['url'],flush=True)
        with tempfile.NamedTemporaryFile(dir=directory,delete=False) as temporary:
            pending=Path(temporary.name)
        try:
            with urlopen(item['url'],timeout=60) as response,pending.open('wb') as stream:
                remaining=item['bytes']+1
                while remaining:
                    block=response.read(min(1024*1024,remaining))
                    if not block:break
                    stream.write(block);remaining-=len(block)
            if pending.stat().st_size!=item['bytes'] or digest(pending)!=item['sha256']:
                raise ValueError(f'Download size/hash mismatch: {item["filename"]}')
            pending.replace(path)
        finally:pending.unlink(missing_ok=True)
    print('Research kernel set ready:',directory.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'data/kernels')
    parser.add_argument('--mission',action='store_true',help='Include Mars body-centre SPK and surface orientation PCK')
    args=parser.parse_args()
    fetch(args.directory,args.mission)
