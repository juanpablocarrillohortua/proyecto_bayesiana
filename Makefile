.PHONY: help clean install install-dev lint lint-py lint-nb format \
        format-check quality test validate scrape scrape-venta

.DEFAULT_GOAL := help

# Prefer the project virtualenv, fall back to whatever is on PATH.
# Override with: make <target> PYTHON=python3
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := .venv/Scripts/python.exe
else
    VENV_PYTHON := .venv/bin/python
endif
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)

# `python -m ruff` rather than bare `ruff`: guarantees the venv's copy and
# sidesteps PATH/.exe resolution on Windows.
RUFF := $(PYTHON) -m ruff

PY_DIRS  := scraper src utils
NB_DIRS  := notebooks
ALL_DIRS := $(PY_DIRS) $(NB_DIRS)

help: ## display this help message
	@$(PYTHON) tools/mk_help.py $(MAKEFILE_LIST)

install: ## install scraper runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## install linting and formatting tooling
	$(PYTHON) -m pip install -r requirements/dev.txt

clean: ## remove byte code, caches and notebook checkpoints
	@$(PYTHON) tools/clean.py

lint-py: ## check PEP 8 in .py files (scraper, src, utils)
	$(RUFF) check $(PY_DIRS)

lint-nb: ## check PEP 8 in notebooks, cell by cell
	$(RUFF) check $(NB_DIRS)

lint: lint-py lint-nb ## check PEP 8 in every source folder

format-check: ## verify layout and whitespace without rewriting files
	$(RUFF) format --check --diff $(ALL_DIRS)

# --exit-zero: unfixable findings (naming, complexity) must not stop the
# formatter from running. Use `make lint` afterwards to see what is left.
format: ## auto-fix imports, layout and safe PEP 8 violations
	$(RUFF) check --fix --exit-zero $(ALL_DIRS)
	$(RUFF) format $(ALL_DIRS)

quality: clean lint format-check ## read-only PEP 8 gate (safe for CI)

test: ## run the offline unit tests (stages 3 and 4)
	$(PYTHON) -m pytest tests -q

validate: quality test ## run all checks

# ---------------------------------------------------------------------
# Scraper. OPERATION/CITY/PAGES are overridable:
#   make scrape PAGES=1-3
#   make scrape OPERATION=venta CITY=bogota
# ---------------------------------------------------------------------
OPERATION ?= arriendo
CITY      ?= bogota
PAGES     ?=
STAGE     ?= all
# Each operation/city needs its own directory: the pipeline refuses to
# mix two queries' artifacts, so sharing ./out would fail on the guard.
OUT       ?= ./out-$(OPERATION)-$(CITY)

scrape: ## run the complete FincaRaiz pipeline (stages 0-6)
	$(PYTHON) -m scraper run --stage $(STAGE) --operation $(OPERATION) \
		--city $(CITY) --out $(OUT) $(if $(PAGES),--pages $(PAGES),)

scrape-venta: ## same pipeline for sale listings
	$(MAKE) scrape OPERATION=venta
