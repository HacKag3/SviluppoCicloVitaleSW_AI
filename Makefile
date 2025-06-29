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
	docker build -t fer2013 .
docker_train: docker
	docker run --rm --name fer2013_trainer \
		-v "$(DOCKER_PWD)/persistent_data:/app/results" \
		fer2013 \
		python src/train_fer2013.py
	docker image prune -f
docker_inference: docker
	docker run --rm --name fer2013_inference \
		-v "$(DOCKER_PWD)/persistent_data:/app/results" \
		fer2013 \
		python src/inference_fer2013.py
	docker logs fer2013_inference
	docker image prune -f