from setuptools import setup, find_packages

setup(
    name="aimakerspace",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.12.0",
        "pandas>=2.2.0",
        "PyPDF2>=3.0.0",
        "pycryptodome>=3.19.0",
    ],
) 