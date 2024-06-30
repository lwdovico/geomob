from setuptools import setup, find_packages

with open('requirements.txt') as reqfile:
    requirements = reqfile.read().splitlines()

with open("README.md", "r") as readmefile:
    long_description = readmefile.read()

setup(
    name = "geomob",
    version = '0.0.3',
    author = "Ludovico Lemma",
    author_email = "ludovico.lemma@mindearth.ch",
    description = "Geospatial and mobility data library",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lwdovico/geomob",
    packages = find_packages(),
    setup_requires = ["wheel"],
    install_requires = requirements,
    classifiers = [
        "Programming Language :: Python :: 3"
    ],
    package_data = {
    }
)