from setuptools import setup, find_packages

setup(
    name="sandbox",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'biopython',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)