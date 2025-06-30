build:
	python -m build

install:
	pip install -e .

lint:
	ruff check src --ignore E731,E741,F405
	ruff format src --diff
	mypy src

lint-fix:
	ruff check src --ignore E731,E741,F405 --fix
	ruff format src

.PHONY: docs

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/qkan/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf qkan.egg-info
	rm -rf src/qkan.egg-info
	rm -rf docs/_build
