.PHONY: build run-train run-inference clean

build:
	docker-compose build

run-train:
	docker-compose run mamba-trainer ./scripts/train.sh

run-prepare:
	docker-compose run mamba-trainer ./scripts/prepare_data.sh

run-inference:
	docker-compose up mamba-inference

clean:
	docker-compose down
	docker system prune -f

	docker-compose up -d

stop:
	docker-compose stop

logs:
	docker-compose logs -f

test:
	docker-compose run mamba-trainer pytest

.PHONY: dev stop logs test
