from setuptools import setup, find_packages

setup(
    name="arxbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "sentence-transformers",
        "chromadb",
        "openai",
        "tiktoken",
        "pyarrow",

    ],
    python_requires='>=3.7',
)