PY ?= python

.PHONY: install test lint demo benchmark tests-gen serve docker

install:
	$(PY) -m pip install -e ".[dev,api]"

test:
	pytest -q

lint:
	ruff check src tests scripts

demo:
	otasentry demo --workdir runs/demo
	@echo "open runs/demo/report/report.html"

tests-gen: demo
	otasentry gen-tests --baseline runs/demo/campaign/pre --findings runs/demo/report/findings.json \
		--out generated/test_ev_platform.py

benchmark:
	$(PY) scripts/benchmark.py --out docs/benchmark.md

serve:
	otasentry serve

docker:
	docker build -t otasentry .
