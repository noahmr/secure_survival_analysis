import setuptools

setuptools.setup(
    name = 'secure_survival_analysis',
    version = '0.1',
    author = 'Noah van der Meer',
    description = 'Privacy-preserving survival analysis using multiparty computation',
    long_description = 'file: README.md',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/noahmr/secure_survival_analysis',
    packages=['secure_survival_analysis'],
    keywords = ['survival analysis', 'multiparty computation'],
    license = "MIT",
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    install_requires = ['mpyc', 'numpy', 'gmpy2'],
    python_requires = '>=3.11'
)