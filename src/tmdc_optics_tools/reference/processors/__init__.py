# processors/__init__.py
from importlib import import_module
from pathlib import Path

for _path in Path(__file__).parent.glob("*.py"):
    if _path.stem != "__init__":
        from importlib import import_module
        _mod = import_module(f".{_path.stem}", package=__name__)
        globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})