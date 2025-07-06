from setuptools import setup, find_packages

setup(
    name="trading-bot",
    version="1.0.0",
    author="David Franco",
    description="Sistema de trading bot con múltiples agentes",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "supabase",
        "yfinance",
        "numpy",
        "pandas",
        "requests",
        "anthropic",
        "alpaca-trade-api",
        "textblob",
        "matplotlib",
        "tabulate",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov",
            "pytest-mock",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-bot=main_bot:main",
        ],
    },
)