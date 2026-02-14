from setuptools import setup, find_packages

setup(
    name="eagleeye",
    version="0.1.0",
    description="AI-powered security intelligence platform for predicting and anticipating armed bandit activities in Northern Nigeria",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "nltk>=3.8.0",
        "geopy>=2.4.0",
        "python-dateutil>=2.8.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "eagleeye=eagleeye.cli:main",
        ],
    },
)
