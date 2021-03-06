# Variables
SOURCE = motel

# Commands
all: lint test
compile: clean pyinstaller
test: unit
publish: test upload clean # TODO: lint is missing

clean:
	rm -rf *.egg-info .pytest_cache build dist
	find . -name "*.pyc" | xargs rm
	find . -name "__pycache__" | xargs rm -rf
	rm -rf ftest/content
	rm -f *.spec

lint:
	@echo Linting source code using pep8...
	pycodestyle --ignore E501,E722 $(SOURCE) test
	@echo

unit:
	@echo Running unit tests...
	pytest -s
	@echo

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*
