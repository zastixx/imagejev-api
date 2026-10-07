"""Setup script for imagejev package (fallback for older pip/setuptools environments)."""
from setuptools import find_packages, setup

setup(
    name="imagejev",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"imagejev": ["py.typed"]},
    include_package_data=True,
)
