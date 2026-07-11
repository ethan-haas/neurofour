# NeuroFour — backend verify & run recipes.
# Position sets and the leaderboard ship pre-generated in bench_data/, so the
# common targets are test / bench / api.

PY ?= python

.PHONY: bench verify test api

# Regenerate bench_data/leaderboard.json from the agent registry (slow: runs
# the full ladder, including the exact solver).
bench:
	$(PY) scripts/run_bench.py

# Regenerate then assert the committed leaderboard is reproducible.
verify:
	$(PY) -m pytest tests/ -q && $(PY) scripts/run_bench.py --check

test:
	$(PY) -m pytest tests/ -q

# Run the API locally on http://localhost:8000
api:
	$(PY) -m uvicorn app.main:app --reload
