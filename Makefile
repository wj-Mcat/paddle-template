PYTHONPATH=./
SOURCE_GLOB=$(wildcard src/**.py tests/**.py examples/**.py)

IGNORE_PEP=E203,E221,E241,E272,E501,F811

.PHONY: all
all : clean test 

.PHONY: clean
clean:
	rm -fr dist/ build/ .pytype


# disable: TODO list temporay
.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		$(SOURCE_GLOB)

.PHONY: pytest
pytest:
	pytest tests


.PHONY: install
install:
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt
	# $(MAKE) install-git-hook

.PHONY: test
test: pylint pytest

.PHONY: version
version:
	@newVersion=$$(awk -F. '{print $$1"."$$2"."$$3+1}' < VERSION) \
		&& echo $${newVersion} > VERSION \
		&& git add VERSION \
		&& git commit -m "🔥 update version to $${newVersion}" > /dev/null \
		&& git tag "v$${newVersion}" \
		&& echo "Bumped version to $${newVersion}"

.PHONY: deploy-version
deploy-version:
	echo "version = '$$(cat VERSION)'" > src/paddle_prompt/version.py

.PHONY: doc
doc:
	mkdocs serve
