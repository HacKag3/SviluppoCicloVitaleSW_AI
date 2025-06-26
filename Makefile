# Scarica tutte le dipendeze per preparare l'ambiente
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo Installazione delle dipendenze terminata.
# Analisi Statistica del codice sorgente
lint:
	PYTHONPATH=. python -m pylint --disable=R,C src/*.py tests/*.py
	@echo Linting complete.
# Unit test
test:
	PYTHONPATH=. python -m pytest -vv --cov=src tests/
	@echo Testing complete.