WHL_FILENAME := $(shell find dist/. -type f -iname "*.whl" -exec basename {} \;)

install-deps:
	poetry install --no-root
	pre-commit install-hooks

local-build:
	rm -rf dist
	poetry build -n

create-requirements:
	poetry export -o requirements.txt --without-hashes

lint:
	poetry run mypy dataqualityreport
	poetry run pre-commit run --all-files

lint-docker:
	docker run --rm -i hadolint/hadolint < Dockerfile

unittest:
	pytest

make notebook:
	poetry run jupyter notebook

clean:
	find . -name \*.pyc -delete
	find . -name \*.pyo -delete
	find . -name \*.cache -exec rm -rf {} +
	find . -name \__pycache__ -exec rm -rf {} +
	find . -name \*.pytest_cache -exec rm -rf {} +
	rm -Rf build
	rm -Rf dist
	rm -Rf *.egg-info

