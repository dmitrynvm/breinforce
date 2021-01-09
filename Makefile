install:
	@pip install -e .

test:
	@pytest

lint:
	@flake8

build: check
	@poetry build

publish:
	@poetry publish

clean:
	@sh clean.sh
