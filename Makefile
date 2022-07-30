clean:
	rm -rf dist
uninstall:
	pip3 uninstall dist/fast_nbconvert-0.1.0-py3-none-any.whl -y
build-native:
	dune build --profile production
build-p: build-native
	poetry build
rename-p:
	rename "s/cp38-cp38-manylinux_2_31_x86_64/py3-none-any/" dist/*

reinstall: uninstall build-p rename-p
	pip3 install dist/*.whl

republish-pypi: clean build-p rename-p
	poetry publish