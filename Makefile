install:
	@sh install.sh

server:
	@python3 breinforce/api/server.py

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
