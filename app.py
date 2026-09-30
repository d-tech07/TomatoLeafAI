"""TomatoLeafAI v21 -- Flask web app (local and Vercel).

    Local:   pip install -r requirements.txt && python app.py      -> http://127.0.0.1:5000
    Vercel:  see README.md (vercel.json, models via bundle or MODEL_BASE_URL)

No TensorFlow at runtime: every model runs through LiteRT (.tflite files produced
by tools/export_litert.py), including Grad-CAM, which is computed inside each
model's exported head graph.
"""
import json
import os
import time

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from tomatoleaf_web.pipeline import ModelStore, TomatoLeafEngine, MODEL_META, pretty
from tomatoleaf_web import treatment as T

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(ROOT, 'models'))
MODEL_BASE_URL = os.environ.get('MODEL_BASE_URL')          # e.g. a GitHub Release download URL
CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.60'))
MAX_UPLOAD_MB = 4                                           # Vercel request bodies are capped at 4.5 MB

# Static files live in public/static so Vercel's CDN serves them at /static/...;
# locally Flask serves the same folder at the same URL.
app = Flask(__name__, static_folder=os.path.join(ROOT, 'public', 'static'), static_url_path='/static',
            template_folder=os.path.join(ROOT, 'templates'))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

_ENGINE = None
_ENGINE_T = 0.0


def engine():
    """Created on first use so a cold start only pays for what a request needs.
    If the model manifest could not be downloaded (network hiccup), retry after 30 s."""
    global _ENGINE, _ENGINE_T
    stale = (_ENGINE is not None and not _ENGINE.available_models() and MODEL_BASE_URL
             and time.time() - _ENGINE_T > 30)
    if _ENGINE is None or stale:
        _ENGINE = TomatoLeafEngine(ModelStore(MODELS_DIR, MODEL_BASE_URL), conf_threshold=CONF_THRESHOLD)
        _ENGINE_T = time.time()
    return _ENGINE


ALLOWED = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def _file_or_error():
    f = request.files.get('image') or request.files.get('leaf_image')
    if f is None or not f.filename:
        return None, 'Choose a photo of one tomato leaf first.'
    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if ext and ext not in ALLOWED:
        return None, 'Unsupported file type — use JPG, PNG, WEBP or BMP.'
    data = f.read()
    if not data:
        return None, 'The uploaded file is empty.'
    return data, None


@app.after_request
def _headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    return resp


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({'error': f'That image is larger than {MAX_UPLOAD_MB} MB. The page shrinks photos '
                             'automatically — try again or use a smaller photo.'}), 413


@app.route('/')
def index():
    e = engine()
    return render_template('index.html', models=e.models_info(), default_model=e.default_model,
                           classes=[{'id': c, 'name': pretty(c)} for c in e.class_names],
                           conf_threshold=CONF_THRESHOLD, status=e.status(), author='Dammar Khadayat')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Streams NDJSON: progress events, one 'partial' per model, then the final 'result'."""
    data, err = _file_or_error()
    if err:
        return jsonify({'error': err}), 400
    model = request.form.get('model') or None
    compare = request.form.get('mode') == 'compare'

    def gen():
        try:
            for ev in engine().analyze(data, model_name=model, compare=compare):
                yield json.dumps(ev, default=float) + '\n'
        except Exception as exc:  # noqa: BLE001 -- report to the page instead of a broken stream
            app.logger.exception('analyze failed')
            yield json.dumps({'type': 'error', 'error': f'Could not analyse this image ({exc}).'}) + '\n'

    return Response(stream_with_context(gen()), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Plain JSON (non-streaming) version of /api/analyze for scripts and API clients."""
    data, err = _file_or_error()
    if err:
        return jsonify({'error': err}), 400
    result = None
    try:
        for ev in engine().analyze(data, model_name=request.form.get('model'),
                                   compare=request.form.get('mode') == 'compare'):
            if ev['type'] == 'result':
                result = ev['result']
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': f'Could not analyse this image ({exc}).'}), 500
    return jsonify(result)


@app.route('/api/models')
def api_models():
    e = engine()
    return jsonify({'models': e.models_info(), 'default_model': e.default_model})


@app.route('/api/diseases')
def api_diseases():
    e = engine()
    out = []
    for c in e.class_names:
        d = dict(T.TREATMENT_DATABASE.get(c, {}))
        d.pop('severity_note', None)
        out.append({'id': c, 'name': pretty(c), **d})
    return jsonify({'classes': out, 'disclaimer': T.DISCLAIMER})


@app.route('/api/performance')
def api_performance():
    """Evaluation files exported next to the models (optional)."""
    s = engine().store
    out = {'manifest_models': s.manifest.get('models', {})}
    for key, fn in (('gate', 'gate_v21_config.json'), ('hypotheses', 'hypothesis_verdicts_v21.json'),
                    ('calibration', 'calibration_summary.json')):
        out[key] = s.read_json(fn)
    for key, fn in (('comparison_csv', 'final_comparison_table.csv'), ('xai_csv', 'xai_v21_summary.csv'),
                    ('latency_csv', 'latency_model_size_v21.csv')):
        out[key] = s.read_text(fn)
    return jsonify(out)


@app.route('/health')
def health():
    t = time.time()
    e = engine()
    st = e.status()
    st.update({'status': 'ok' if st['models'] else 'models_missing', 'models_dir': MODELS_DIR,
               'model_base_url': bool(MODEL_BASE_URL), 'init_ms': int((time.time() - t) * 1000)})
    return jsonify(st)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=port, debug=False, threaded=True)
