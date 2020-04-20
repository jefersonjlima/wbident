.PHONY: help prepare-dev test lint run clean docker

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3

.DEFAULT: help
help:
	@echo "make prepare-dev"
	@echo "       prepare development environment"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       run pylint and mypy"

prepare-dev:
	apt-get -y install python3 python3-pip python3-tk
	python3 -m pip install virtualenv
	make venv

venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: setup.py
	test -d $(VENV_NAME) || virtualenv -p python3 $(VENV_NAME)
	${PYTHON} -m pip install -U pip setuptools
	${PYTHON} -m pip install -e .  --upgrade --ignore-installed
	touch $(VENV_NAME)/bin/activate

test: venv
	${PYTHON} -m pytest -vv tests

lint: venv
	${PYTHON} -m pylint
	${PYTHON} -m mypy --ignore-missing-imports quore

clean:
	@rm -rf $(VENV_NAME) *.eggs *.egg-info .cache .mypy_cache/ .pytest_cache/ ecg_api.log

