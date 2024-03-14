#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup

DESCRIPTION = 'Open source Python library converting pdf to docx.'
EXCLUDE_FROM_PACKAGES = ["build", "dist", "test"]

# read version number from version.txt, otherwise alpha version
# Github CI can create version.txt dynamically.
def get_version(fname):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            version = f.readline().strip()
    else:
        version = 'alpha-1.0.0'

    return version

# Load README.md for long description
def load_long_description(fname):
    if os.path.exists(fname):
        with open(fname, 'r', encoding="utf-8") as f:
            long_description = f.read()
    else:
        long_description = DESCRIPTION

    return long_description

def load_requirements(fname):
    try:
        # pip >= 10.0
        from pip._internal.req import parse_requirements        
    except ImportError:
        # pip < 10.0
        from pip.req import parse_requirements

    reqs = parse_requirements(fname, session=False)
    try:
        requirements = [str(ir.requirement) for ir in reqs]
    except AttributeError:
        requirements = [str(ir.req) for ir in reqs]

    return requirements

setup(
    name="pdfparser",    
    version=get_version("version.txt"),
    keywords=["pdf-parser", "pdf-to-text"],
    description=DESCRIPTION,
    long_description=load_long_description("README.md"),
    long_description_content_type="text/markdown",
    license="GPL v3", 
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,    
    zip_safe=False,
    install_requires=load_requirements("requirements.txt"),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "pdfparser=pdfparser.main:main"
            ]},
)