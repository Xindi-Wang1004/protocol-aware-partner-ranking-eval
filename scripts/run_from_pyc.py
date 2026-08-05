#!/usr/bin/env python3
"""Run evaluation modules from scripts/__pycache__ when .py sources are missing."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from importlib.machinery import SourcelessFileLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYCACHE = PROJECT_ROOT / "scripts" / "__pycache__"


def load_pyc_module(name: str):
    pyc = PYCACHE / f"{name}.cpython-310.pyc"
    if not pyc.exists():
        raise FileNotFoundError(f"Missing bytecode cache: {pyc}")
    loader = SourcelessFileLoader(f"scripts.{name}", str(pyc))
    spec = importlib.util.spec_from_loader(f"scripts.{name}", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"scripts.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_scripts_package() -> types.ModuleType:
    pkg = sys.modules.get("scripts")
    if pkg is None:
        pkg = types.ModuleType("scripts")
        pkg.__path__ = [str(PROJECT_ROOT / "scripts")]
        sys.modules["scripts"] = pkg
    return pkg


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "Usage: run_from_pyc.py <module> [args...]\n"
            "Modules: constrained_2104_eval, constrained_2104_diagnostics, necessity_eval"
        )
        raise SystemExit(0 if argv and argv[0] in {"-h", "--help"} else 1)

    module_name = argv[0]
    module_args = argv[1:]
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    pkg = ensure_scripts_package()

    # dependency order for constrained_2104_eval / diagnostics
    deps: list[str] = []
    if module_name in {"constrained_2104_eval", "constrained_2104_diagnostics"}:
        deps.extend(["retrieval_eval_utils", "necessity_eval"])
    if module_name == "constrained_2104_diagnostics":
        deps.append("constrained_2104_eval")
    for dep in deps:
        if f"scripts.{dep}" not in sys.modules:
            loaded = load_pyc_module(dep)
            setattr(pkg, dep, loaded)

    mod = load_pyc_module(module_name)
    setattr(pkg, module_name, mod)

    # Bytecode was compiled with PROJECT_ROOT=scripts/; patch to repo root.
    for name in ("constrained_2104_eval", "constrained_2104_diagnostics", "necessity_eval"):
        m = sys.modules.get(f"scripts.{name}")
        if m is not None and hasattr(m, "PROJECT_ROOT"):
            m.PROJECT_ROOT = PROJECT_ROOT

    old_argv = sys.argv[:]
    try:
        sys.argv = [f"scripts/{module_name}.py", *module_args]
        if hasattr(mod, "main"):
            mod.main()
        else:
            raise RuntimeError(f"Module {module_name} has no main()")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
