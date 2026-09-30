# TomatoLeafAI — Web App (v21, Flask + Vercel)

Photograph a tomato leaf → background removal → "is this a tomato leaf?" gate → diagnosis with the
proposed **LAG-HViT** hybrid or any baseline (or all five side by side) → Grad-CAM, Leaf-Focus Score and
disease regions → treatment and management advice.

Author: Dammar Khadayat · MIT in Artificial Intelligence, Gandaki University

---

## 1. What is inside

```
app.py                     Flask app (Vercel entrypoint: top-level `app`)
tomatoleaf_web/
  pipeline.py              TomatoLeafEngine: gate → classify → explain → treatment; ModelStore (local or URL)
  litert_engine.py         Runs the exported .tflite models (LiteRT)
  seg_v21.py               Background removal V3 (keeps lesions, removes soil / stones / grass / shadows)
  gate_v21.py              Tomato-leaf gate (shape, serrated edge, veins, colour, texture + deep scores)
  ood_v21.py               Deep tomato-identity check (Relative Mahalanobis + energy + OOD-gate CNN)
  explain.py               Grad-CAM upsampling, Leaf-Focus / Lesion-Focus scores, disease regions
  treatment.py             Treatment database (same content as notebook Cell 11.1) + disclaimer
  ndi_compat.py            OpenCV replacements for the few SciPy functions used (keeps the bundle small)
templates/index.html       Page (Diagnose · History · Models · Diseases · About)
public/static/             app.js, style.css, favicon.svg  (served by Vercel's CDN at /static/…)
tools/export_litert.py     Converts the trained .keras models to .tflite (run once, needs TensorFlow)
tools/keras_layers_v21.py  The v21 custom layers (only the export tool imports this)
tests/test_smoke.py        Smoke test
vercel.json · .vercelignore · requirements.txt · .python-version
```

**Why no TensorFlow on the server?** `tensorflow-cpu` alone is ~1.2 GB installed, which is over
Vercel's 500 MB function limit. The export tool converts every model to LiteRT (`.tflite`, float16
weights), and each model's Grad-CAM is computed *inside* the exported graph, so the server only needs
Flask + NumPy + Pillow + OpenCV + LiteRT (~320 MB installed, measured for Python 3.12 on Linux).
Checked on test models: LiteRT vs Keras probabilities differ by ≤ 0.00006 and normalised Grad-CAM by ≤ 0.002.

Models served: `Hybrid_CNN_ViT` (LAG-HViT, proposed), `EfficientNetB7`, `VGG16`, `RegNetY008`,
`CNN_Only`, and the ablations `NaiveConcat_CNN_ViT`, `ViT_Branch`, `Hybrid_CNN_ViT_noCAMreg` if you export them.

---

## 2. Export the trained models (once, on the machine that trained them)

After the v21 notebook has finished, you have `models/<Name>_final.keras`, `models/ood_gate.keras`,
`models/feature_ood_v21.json` and a `results/` folder.

```bash
# in a separate environment with the SAME TensorFlow/Keras version as training
pip install -r requirements-export.txt
python tools/export_litert.py --models-dir /path/to/models --results-dir /path/to/results --out models_litert
```

This writes `models_litert/` (`<Name>.trunk.tflite`, `<Name>.head.tflite`, `ood_gate.tflite`, the gate /
OOD / calibration JSON files, the result CSVs for the Models tab, and `manifest.json` with SHA-256
checksums). Each model is checked against Keras and the difference is printed.

Options: `--only Hybrid_CNN_ViT VGG16` (subset) · `--int8` (about half the size of float16; probabilities
stay very close, but Grad-CAM maps drift more, so check the printed verification) · `--no-fp16` (float32).

Expected size with float16: roughly 350 MB for the five main models (EfficientNetB7 and the hybrid are the largest).

---

## 3. Run locally

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py                      # → http://127.0.0.1:5000
```

If `conda` is also installed and you see `_distutils_hack` errors, run `conda deactivate` first (until the
prompt shows no `(base)`), then create the venv again.

**macOS / Linux**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Put `models_litert/` next to `app.py` (or set `MODELS_DIR`). Smoke test: `python tests/test_smoke.py path/to/leaf.jpg`.

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` | `./models_litert` | Folder with the exported models |
| `MODEL_BASE_URL` | – | URL the model files are downloaded from when they are not in `MODELS_DIR` |
| `CONF_THRESHOLD` | `0.60` | Below this calibrated confidence the app says it is not sure |
| `PORT`, `HOST` | `5000`, `127.0.0.1` | Local server only |

---

## 4. Deploy to Vercel

Vercel runs the Flask app as a single Python function (Fluid compute). The limits that matter here:

| Limit | Value | How the app handles it |
|---|---|---|
| Function bundle | 500 MB (Large Functions beta: up to 5 GB) | Code + dependencies ≈ 320 MB; models go to `/tmp` (Option A) or use Large Functions (Option B) |
| `/tmp` | 500 MB writable | Models downloaded once per instance, SHA-256 checked |
| Request body | 4.5 MB | The page resizes photos in the browser to ≤ 1280 px JPEG (< 3.5 MB); server cap 4 MB |
| Max duration | 300 s Hobby (800 s Pro) | `maxDuration: 300` in `vercel.json`; a diagnosis takes ~1 s warm |
| Memory | 2 GB Hobby (up to 4 GB Pro) | All five models loaded: about 1 GB (estimate) |
| Static files | Only from `public/**` | CSS/JS/icon are in `public/static/` |

### Option A — models on a GitHub Release (recommended, works on the free Hobby plan)

1. Push this folder to a GitHub repository (without `models_litert/`; `.gitignore` and `.vercelignore` handle it).
   GitHub rejects files over 100 MB in normal commits, which is why the models go to a *Release*.
2. Create a Release and attach every file in `models_litert/` (release assets can be up to 2 GB each):
   ```bash
   gh release create models-v21 models_litert/* --title "TomatoLeafAI models v21" --notes "LiteRT export"
   ```
   The repository (or a separate public repo used only for the models) must be **public** so Vercel can
   download without a token. Hugging Face also works: `https://huggingface.co/<user>/<repo>/resolve/main`.
3. On vercel.com → **Add New… → Project** → import the repo. Framework preset: detected as Flask (or "Other").
4. **Settings → Environment Variables**:
   `MODEL_BASE_URL = https://github.com/<user>/<repo>/releases/download/models-v21`
5. Deploy. Open `https://<project>.vercel.app/health`; it should show `"status": "ok"` and the model list.

The first diagnosis on a new instance downloads only the model it needs (plus `ood_gate.tflite`),
which adds a few seconds; later requests on that instance reuse `/tmp`. "Compare all" downloads the rest once.

### Option B — bundle the models in the function (Large Functions, public beta)

1. Delete the `models_litert/` line in `.vercelignore` and keep `models_litert/` next to `app.py`.
2. Large Functions are on by default for new projects on Fluid compute; for an existing project add the
   environment variable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.
3. Deploy with the CLI (the model files are too large for a normal Git push):
   ```bash
   npm i -g vercel
   vercel login
   vercel --prod
   ```
   No `MODEL_BASE_URL` is needed. Cold starts are slower (bigger bundle), but no download happens.

### Local test in the Vercel runtime (optional)

```bash
vercel dev        # uses vercel.json and public/, like production
```

---

## 5. API

| Method | Path | Returns |
|---|---|---|
| POST | `/api/analyze` | NDJSON stream: `progress` events, one `partial` per model, then `result`. Form fields: `image` (file), `mode` = `single`/`compare`, `model` |
| POST | `/api/predict` | Same result as one JSON object (for scripts) |
| GET | `/api/models` | Installed models with labels, size, parameters, calibration |
| GET | `/api/diseases` | All 10 classes with treatment advice + disclaimer |
| GET | `/api/performance` | Exported notebook results (comparison table, gate rates, hypotheses, XAI, latency) |
| GET | `/health` | Status, installed models, gate mode |

```bash
curl -F image=@leaf.jpg -F mode=single https://<project>.vercel.app/api/predict
```

The `result` contains: `accepted`, `message`, `reasons` (why a photo was rejected), `preview`
(background-removed image and mask), `prediction` (class, calibrated confidence, top-5, Grad-CAM /
disease-region / lesion-attention images, Leaf-Focus Score, Lesion-Focus Score), `results` (per model
in compare mode), `consensus`, `affected_area_estimate` (share of leaf area with lesion colour — an
estimate, not a severity grade), `treatment` and `timing_ms`.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: werkzeug` / `flask` | The venv is not active or `pip install -r requirements.txt` was run in a different Python. Activate `.venv` and reinstall. |
| `tensorflow-cpu==… not found` | Not needed any more: the web app uses `requirements.txt` (no TensorFlow). TensorFlow is only for `tools/export_litert.py`. |
| Page says "No models installed" | `models_litert/` missing next to `app.py`, or `MODEL_BASE_URL` not set / not public. Check `/health`. |
| `/health` shows `manifest download failed` | `MODEL_BASE_URL` is wrong or the Release is private. Open `<MODEL_BASE_URL>/manifest.json` in a browser. |
| Vercel build fails: bundle over 500 MB | Models were uploaded (Option B without Large Functions). Use Option A, or enable Large Functions. |
| `413` error | Photo over 4 MB was posted directly to the API; the web page resizes automatically. |
| Every photo "Not accepted" | Photograph one leaf filling most of the frame, in focus, in daylight. The reasons are listed under "Why?". |

The advice is decision support, not a substitute for a local agronomist or the pesticide label.
