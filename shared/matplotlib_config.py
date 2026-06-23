import os
import tempfile


def configure_matplotlib() -> None:
    """Use a writable, headless Matplotlib setup for Dash callbacks."""
    cache_dir = os.path.join(tempfile.gettempdir(), "tactiq_matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)

    import matplotlib

    matplotlib.use("Agg", force=True)

