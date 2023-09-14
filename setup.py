import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kamui",
    version="0.0.1",
    author="Chin-Yun Yu",
    author_email="chin-yun.yu@qmul.ac.uk",
    description="A Python package for unwrapping phase",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/kamui",
    packages=["kamui"],
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)