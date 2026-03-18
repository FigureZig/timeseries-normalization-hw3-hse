.PHONY: install test run analyze clean all

install:
	pip install -r requirements.txt

test:
	python run_experiment.py --n_series 5
	python analysis_results.py

run:
	python run_experiment.py

analyze:
	python analysis_results.py

clean:
	rm -rf results/*
	rm -rf data/*.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install run analyze