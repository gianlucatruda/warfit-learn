import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="warfit-learn-gianlucatruda",
    version="0.0.1",
    author="Gianluca Truda",
    author_email="gianlucatruda@gmail.com",
    description="A toolkit for reproducible research in warfarin dose estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gianlucatruda/warfit-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
)
