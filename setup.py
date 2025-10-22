# oxide_analysis_general_package/setup.py
from setuptools import setup, find_packages

# oxide_analysis_general_package/setup.py
from setuptools import setup, find_packages

setup(
    name="oxide_analysis_general",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20,<1.24",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "tqdm>=4.60",
        "openpyxl>=3.0",
        "scipy>=1.7",
        "scikit-optimize==0.9.0",
        "scikit-learn>=1.3",
        "xgboost>=1.7,<2.0",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'oxide-analysis-general=oxide_analysis_general.main:main',
        ],
    },
    author="Lulu Li",
    author_email="lli@iciq.es",
    description="Generalized Oxide Descriptor Analysis Package",
)
