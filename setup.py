from setuptools import setup, find_packages

setup(
    name="ssb",
    version="0.1",
    packages=find_packages(),
    # install_requires=[
    #     "torch", 
    #     "requests", 
    #     # any other dependencies your package has
    # ],
    install_requires=[],
    package_data={
        'SSB': ['splits/*.json'],
    },
    author="Sagar Vaze",
    author_email="sagar@robots.ox.ac.uk",
    description="Download and dataloader utilities for the SSB benchmark suite",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sgvaze/SSB",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)