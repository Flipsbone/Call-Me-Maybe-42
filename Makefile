PYTHON = uv run python3 
MAIN = -m src
SRC = src/

all: install

install: .venv/uv.lock
.venv/uv.lock: pyproject.toml
	@echo "Installing dependencies using uv..."
	uv lock --check || uv lock
	uv sync
	@touch .venv/uv.lock

run: install
	@echo "Running the program..."
	mkdir -p data/output
	$(PYTHON) $(MAIN) $(ARGS) 

debug: install
	@echo "Starting debug mode..."
	$(PYTHON) -m pdb $(MAIN) $(ARGS)

lint: install
	@echo "Running standard linting..."
	uv run flake8 $(SRC)
	uv run mypy $(SRC)

lint-strict: install
	@echo "Running strict linting..."
	uv run flake8 $(SRC)
	uv run mypy --strict $(SRC)

clean:
	@echo "Cleaning up..."
	rm -rf .mypy_cache \
	       .pytest_cache \
		   .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: all install run debug lint lint-strict clean

