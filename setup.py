from setuptools import setup, find_packages

setup(
    name="news-insight-app",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Flask==2.3.3",
        "requests==2.31.0",
        "textblob==0.17.1",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.2",
            "pytest-cov==4.1.0",
            "pytest-flask==1.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "news-insight-app=news_insight_app.main:main",
        ],
    },
)