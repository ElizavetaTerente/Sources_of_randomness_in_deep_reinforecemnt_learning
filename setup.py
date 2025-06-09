from setuptools import find_packages
from setuptools import setup

setup(
    name="oracle",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "superOracle=oracle.superOracle:main",
            "oracle=oracle.oracle:main",
            "PPOscript=oracle.PPOscript:main",
            "SACscript=oracle.SACscript:main",
        ],
    },
)
