from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List

import numpy as np

IN_BROWSER = find_spec("js") is not None


_ort = None

if not IN_BROWSER:
    try:
        import onnxruntime as _ort
    except ImportError:
        pass


async def patch_onnx():
    global _ort
    import js

    _ort = await js.eval(
        "import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.bundle.min.mjs')"
    )
    _ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/"


class InferenceSession:
    def __init__(self, path: Path | str):
        path = Path(path)
        if IN_BROWSER:
            if _ort is None:
                raise RuntimeError("Run await patch_onnx() first")
            import js
            from pyodide.ffi import to_js

            data_path = Path(str(path) + ".data")
            opts = js.Object.new()
            opts.executionProviders = to_js(["wasm"])
            if data_path.exists():
                entry = js.Object.new()
                entry.path = data_path.name
                entry.data = to_js(data_path.read_bytes())
                arr = js.Array.new()
                arr.push(entry)
                opts.externalData = arr
            self._js_session = _ort.InferenceSession.create(
                to_js(path.read_bytes()), opts
            )
        else:
            if _ort is None:
                raise RuntimeError(
                    "onnxruntime is not installed. You can use `pip install sketchy[onnx]` to install it\nIf you use `uv` with the git repository, use `uv sync --extra onnx`"
                )
            self._session = _ort.InferenceSession(str(path))

    async def run(
        self, output_names: List[str], input_dict: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        if IN_BROWSER:
            from pyodide.ffi import to_js

            session = await self._js_session
            feeds = {}
            for name, arr in input_dict.items():
                data = to_js(arr.flatten(order="C").astype(np.float32))
                shape = to_js(list(arr.shape))
                feeds[name] = _ort.Tensor.new("float32", data, shape)
            results = await session.run(to_js(feeds))
            return [
                np.array(getattr(results, name).data.to_py(), dtype=np.float32).reshape(
                    list(getattr(results, name).dims.to_py())
                )
                for name in output_names
            ]
        else:
            result = self._session.run(output_names, input_dict)
            return [r[0] if len(r.shape) == 4 else r for r in result]
