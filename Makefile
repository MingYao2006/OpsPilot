# OpsPilot 常用命令（用 .RECIPEPREFIX 避免制表符问题）
.RECIPEPREFIX = >
.PHONY: install run seed test eval docker clean

install:
> python -m pip install -e ".[dev]"

run:
> uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

seed:
> python scripts/seed.py

test:
> python -m pytest -q

eval:
> python scripts/evaluate.py

docker:
> docker compose --profile vector up -d --build

clean:
> python -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in ['work/state','.pytest_cache','app/__pycache__']]"
