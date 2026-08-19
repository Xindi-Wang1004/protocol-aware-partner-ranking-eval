from setuptools import find_packages, setup

setup(
    name="partner-rank-eval",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    entry_points={"console_scripts": ["partner-rank-eval=partner_rank_eval.cli:main"]},
)
