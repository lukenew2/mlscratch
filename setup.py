from setuptools import setup, find_namespace_packages
import pathlib

__version__ = '0.1.0'

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='mlscratch',  
    version=__version__,  
    description='Machine learning from scratch', 
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/lukenew2/mlscratch', 
    author='Luke Newman', 
    author_email='lukeanewman2@gmail.com',  
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='data science, machine learning',  
    packages=find_namespace_packages(include='mlscratch'), 
    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'numpy',
        'scikit-learn'
    ],  
    project_urls={  
        'Source': 'https://github.com/lukenew2/mlscratch',
    },
)
