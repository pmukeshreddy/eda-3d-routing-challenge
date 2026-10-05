"""Minimal in-place C++17 build using the current Python's ABI and headers."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "wire_deps"))
try:
    import pybind11
except ImportError:
    raise SystemExit("Install headers first: python3 -m pip install --target build/wire_deps pybind11==3.0.1")

output = ROOT / "m3d" / ("_wire_native" + sysconfig.get_config_var("EXT_SUFFIX"))
command = shlex.split(os.environ.get("CXX", "c++")) + [
    "-O3", "-std=c++17", "-pedantic-errors", "-Wall", "-Wextra", "-shared", "-fPIC",
    "-fvisibility=hidden", "-I" + pybind11.get_include(),
    "-I" + sysconfig.get_path("include"), str(ROOT / "native" / "wire_engine.cpp"),
    "-o", str(output),
]
if sys.platform == "darwin":
    command += ["-undefined", "dynamic_lookup"]
elif sys.platform == "win32":
    raise SystemExit("This minimal build supports macOS/Linux; Windows needs a compiler-specific build command.")
subprocess.run(command, check=True)
print(f"Built {output}")
