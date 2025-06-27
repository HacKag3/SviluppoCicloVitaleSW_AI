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

ifeq ($(OS),Windows_NT)
    DOCKER_PWD := $(subst \,/,${CURDIR})
else
    DOCKER_PWD := $(CURDIR)
endif
docker:
	docker build -t fer2013_trainer .
	docker run --rm -it --name fer2013_trainer -v "$(DOCKER_PWD)/persistent_data:/app/results" fer2013_trainer
	docker image prune -f