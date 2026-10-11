import sys

print("interpreter:", sys.executable)
try:
    import fastapi

    print("fastapi OK:", fastapi.__version__)
except Exception as exc:  # noqa: BLE001
    print("fastapi FAIL:", exc)
