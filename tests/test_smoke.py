"""Smoke test for the web app (no pytest needed):  python tests/test_smoke.py [path/to/leaf.jpg]

Works without models (checks pages/APIs load and that diagnosis reports "models missing" cleanly);
with models_litert/ present (or MODELS_DIR / MODEL_BASE_URL set) it also runs a full diagnosis.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np            # noqa: E402
from PIL import Image         # noqa: E402
import app as A               # noqa: E402


def jpeg(path=None):
    if path:
        return open(path, 'rb').read()
    rng = np.random.default_rng(0)                           # synthetic green blob on soil
    img = np.full((384, 512, 3), (120, 85, 55), np.uint8) + rng.integers(0, 25, (384, 512, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:384, :512]
    leaf = ((xx - 256) / 150) ** 2 + ((yy - 192) / 110) ** 2 < 1
    img[leaf] = (60, 140, 50)
    b = io.BytesIO(); Image.fromarray(img).save(b, 'JPEG'); return b.getvalue()


def main():
    c = A.app.test_client()
    h = c.get('/health').json
    print('health:', h['status'], '| models:', h['models'])
    assert c.get('/').status_code == 200
    assert c.get('/static/app.js').status_code == 200
    d = c.get('/api/diseases').json
    assert len(d['classes']) == 10 and d['disclaimer']
    print('diseases: 10 classes with treatment advice')
    r = c.post('/api/analyze', data={'image': (io.BytesIO(jpeg(sys.argv[1] if len(sys.argv) > 1 else None)), 'leaf.jpg'),
                                     'mode': 'single'}, content_type='multipart/form-data')
    events = [json.loads(line) for line in r.data.decode().splitlines() if line.strip()]
    print('stream:', [e.get('stage', e['type']) for e in events])
    final = events[-1]
    if final['type'] == 'result':
        res = final['result']
        print('accepted:', res['accepted'], '|', res.get('message', '')[:100])
        if res['accepted']:
            p = res['prediction']
            print('prediction:', p['display_name'], f"{p['confidence']:.1%}", '| LFS', p.get('focus', {}).get('lfs'))
    else:
        print('error event:', final.get('error'))
        assert not h['models'], 'models are installed but analysis failed'
    print('SMOKE TEST OK')


if __name__ == '__main__':
    main()
