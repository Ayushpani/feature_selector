from setuptools import setup, find_packages

setup(
    name="prunex",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
        "plotly>=5.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "playwright>=1.15.0",
        "nest-asyncio>=1.5.0",
    ],
    extras_require={"test": ["pytest>=6.0.0"]},
    author="PruneX Team",
    description="A professional, enterprise-grade feature selection and engineering pipeline.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ayushpani/feature_selector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
