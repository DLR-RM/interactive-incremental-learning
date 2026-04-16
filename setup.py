import os

from setuptools import find_packages, setup

with open(os.path.join("interactive_incremental_learning", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="interactive_incremental_learning",
    packages=[package for package in find_packages() if package.startswith("interactive_incremental_learning")],
    package_data={"interactive_incremental_learning": ["data/*.pickle", "version.txt"]},
    install_requires=[
        "numpy>=1.20,<3.0",
        "scipy>=1.0,<3.0",
        "scikit-learn>=1.0,<3.0",
        "matplotlib>=3.0,<4.0",
        "seaborn>=0.12,<1.0",
    ],
    extras_require={
        "tests": [
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            "mypy",
            "ruff>=0.9.0",
            "black>=25.1.0,<26",
        ]
    },
    description=(
        "Implementation of RA-L paper Interactive incremental learning"
        " of generalizable skill with local trajectory modulation"
    ),
    author="Markus Knauer",
    url="https://github.com/DLR-RM/interactive-incremental-learning",
    author_email="markus.knauer@dlr.de",
    keywords="interactive learning, incremental learning, continual learning, robot learning",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.10",
    project_urls={
        "Code": "https://github.com/DLR-RM/interactive-incremental-learning",
        "RA-L": "https://ieeexplore.ieee.org/document/10887119/",
        "Arxiv": "https://arxiv.org/abs/2409.05655",
        "YouTube": "https://youtu.be/nqigz0l1syA",
    },
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
