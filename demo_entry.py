"""
Streamlit Community Cloud entrypoint for the LyricMood demo.

WHY THIS FILE EXISTS: the demo was going to live on a Hugging Face Space
(docker/Dockerfile.spaces — the single-container image that runs uvicorn + the
Streamlit UI + an embedded file-based Qdrant). In July 2026 HF Spaces' free tier
dropped the Docker/Gradio SDKs, so that plan died. Streamlit Community Cloud
(share.streamlit.io) is the new host: it deploys a branch of the public GitHub
repo by running `streamlit run <entrypoint>` against the repo-root
`requirements.txt`. There is no container, no compose stack, and no separate
Qdrant service — so this entrypoint reconstructs the whole runtime in one
Streamlit process:

  1. fetch the demo bundle (model artifacts + file-based Qdrant) from a public
     HF *model* repo at first boot (cached to disk by huggingface_hub);
  2. point the LYRICMOOD_* settings at the bundle, rewriting the bundled
     registry.json so its container-relative `dir` becomes an absolute path;
  3. start the FastAPI app in-process on 127.0.0.1:8000 (daemon thread) and
     wait for /health;
  4. hand off to the unmodified app/streamlit_app.py UI, which talks to that
     in-process API over HTTP exactly as it does locally.

The Dockerfile.spaces image is kept in the repo as the paid-tier / self-host
option (see docs/DEPLOY_DEMO.md); this file is only for the free SCC host.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I designed the pivot (SCC host, in-process API + HF-model-repo bundle download,
runtime registry rewrite, single-source UI hand-off); Claude wrote the glue.
See ATTRIBUTION.md.
"""

import json
import os
import runpy
import threading
from pathlib import Path

import httpx
import streamlit as st

# --- constants ----------------------------------------------------------------

DEFAULT_BUNDLE_REPO = "evelindsayyy/lyricmood-demo"
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"
HEALTH_TIMEOUT_S = 120.0  # model loading takes ~10-20s; download precedes this
UI_PATH = Path(__file__).parent / "app" / "streamlit_app.py"


# --- bundle acquisition -------------------------------------------------------


def _resolve_bundle_dir() -> Path:
    """Return the local demo-bundle directory.

    LYRICMOOD_DEMO_BUNDLE_DIR short-circuits the download (used for local
    end-to-end testing against the already-built ./demo bundle). Otherwise the
    bundle is pulled from the HF model repo named by LYRICMOOD_DEMO_BUNDLE_REPO
    (default DEFAULT_BUNDLE_REPO) and cached to disk by huggingface_hub.
    """
    local = os.environ.get("LYRICMOOD_DEMO_BUNDLE_DIR")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise FileNotFoundError(f"LYRICMOOD_DEMO_BUNDLE_DIR is not a directory: {path}")
        return path

    from huggingface_hub import snapshot_download

    repo_id = os.environ.get("LYRICMOOD_DEMO_BUNDLE_REPO", DEFAULT_BUNDLE_REPO)
    return Path(snapshot_download(repo_id=repo_id, repo_type="model"))


def _write_runtime_registry(bundle_dir: Path) -> Path:
    """Rewrite the bundled registry so each model `dir` is an absolute path
    inside this bundle, and return the path to the rewritten copy.

    The bundle's registry.json pins `dir: demo/models/transformer` — a path
    relative to the container's working directory that build_demo_bundle.py
    baked in. That path does not exist outside the container, so we rewrite each
    `dir` to `<bundle>/models/<basename>` (absolute) and dump it next to the
    bundle as registry_runtime.json. ModelSpec.dir is a Path, so absolute is
    fine.
    """
    src = bundle_dir / "models" / "registry.json"
    registry = json.loads(src.read_text(encoding="utf-8"))
    for spec in registry.get("models", {}).values():
        if "dir" in spec:
            basename = Path(spec["dir"]).name
            spec["dir"] = str((bundle_dir / "models" / basename).resolve())
    dst = bundle_dir / "registry_runtime.json"
    dst.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return dst


def _configure_environment(bundle_dir: Path) -> None:
    """Point the LYRICMOOD_* settings at the bundle. MUST run before api.main is
    imported (Settings() is constructed at import time via the module-level
    `app = create_app()` in api/main.py). We only import api.main lazily inside
    the server thread (uvicorn's "api.main:app" string form defers it), so
    setting these here — before the thread starts — is in time.
    """
    models_dir = bundle_dir / "models"
    os.environ["LYRICMOOD_MODEL_DIR"] = str(models_dir)
    os.environ["LYRICMOOD_REGISTRY_PATH"] = str(_write_runtime_registry(bundle_dir))
    os.environ["LYRICMOOD_EMBEDDER_DIR"] = str(models_dir / "embedder")
    os.environ["LYRICMOOD_QDRANT_PATH"] = str(bundle_dir / "qdrant_local")
    # One shared rate-limit bucket: every SCC visitor reaches the in-process API
    # from localhost, so the default 30/min would throttle the whole demo at
    # ~15 analyses/min (predict + similar per click). 240/min keeps it usable.
    os.environ["LYRICMOOD_RATE_LIMIT"] = "240/minute"
    os.environ["LYRICMOOD_API_URL"] = API_URL


# --- in-process API -----------------------------------------------------------


def _start_api_thread() -> None:
    """Start the FastAPI app in-process on a daemon thread. Uses uvicorn's
    "api.main:app" string form so api.main is imported inside the thread — after
    _configure_environment has already set the env vars this process reads."""
    import uvicorn

    config = uvicorn.Config(
        "api.main:app", host=API_HOST, port=API_PORT, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="lyricmood-api")
    thread.start()


def _wait_for_health() -> None:
    """Poll /health until the API answers 200, up to HEALTH_TIMEOUT_S. On
    timeout, surface the failure and stop the app (this runs before the UI's
    st.set_page_config, so st.error/st.stop here never conflict with it)."""
    import time

    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{API_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                return
            last_err = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            last_err = type(exc).__name__
        time.sleep(1.0)
    st.error(
        f"LyricMood API did not become healthy within {HEALTH_TIMEOUT_S:.0f}s "
        f"(last: {last_err}). Reload to retry."
    )
    st.stop()


# --- boot (once per SCC container) --------------------------------------------


@st.cache_resource(show_spinner=False)
def boot() -> str:
    """Fetch the bundle, wire the environment, start the API, wait for health.

    Cached so it runs exactly once per container. show_spinner=False keeps this
    from emitting any element before the UI's st.set_page_config (which must be
    the first Streamlit command). Returns the bundle path for cache identity.
    """
    bundle_dir = _resolve_bundle_dir()
    _configure_environment(bundle_dir)
    _start_api_thread()
    _wait_for_health()
    return str(bundle_dir)


boot()

# Hand off to the unmodified UI. runpy keeps app/streamlit_app.py single-source
# (no copy of its body here); it reads LYRICMOOD_API_URL at import, already set
# by boot(), and its st.set_page_config is the first Streamlit command to run.
runpy.run_path(str(UI_PATH), run_name="__main__")
