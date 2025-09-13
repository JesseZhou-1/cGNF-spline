from setuptools import setup, find_packages

setup(
    name='cGNF_spline',
    version='0.0.3',  # start with a small number and increment it with every change
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'networkx',
        'scikit-learn',
        'causalgraphicalmodels',
        'UMNN',
        'joblib',
        'nflows',
    ],
    author='cGNF-Team',
    author_email='cgnf.team@gmail.com',
    description='Causal Graph Normalizing Flows with SplineNormalizer',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license='BSD License',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

