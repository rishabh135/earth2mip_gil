# stages called by modulus ci library
# setup-ci black interrogate lint license install pytest coverage doctest
# https://gitlab-master.nvidia.com/modulus/modulus-launch/-/blob/main/Makefile
# ALL THESE TARGETS NEED to be here for blossom-ci

install:
	git submodule update --init --recursive && \
	pip install --upgrade pip && \
	cd third_party/modulus && pip install . && cd ../.. && \
	pip install -e .

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	echo "TODO"
	true

lint:
	pre-commit run check-added-large-files -a
	pre-commit run flake8 -a

license:
	echo "TODO"
	true

doctest:
	echo "TODO"
	true

pytest:
	coverage run \
		--rcfile='tests/coverage.pytest.rc' \
		-m pytest --ignore=third_party

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*tests* --fail-under=20 && \
		coverage html