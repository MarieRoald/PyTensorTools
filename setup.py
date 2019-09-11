from setuptools import find_packages, setup

setup(
    name="TensorKit-tools",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
