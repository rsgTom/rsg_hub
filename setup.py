from setuptools import setup, find_packages

setup(
    name='brief_scraper',
    version='0.1.0',
    author='Thomas Bowman',
    author_email='thomas@resolutestrategy.com',
    description='A web scraping and data cleaning project for RSG blog posts.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rsgTom/brief_scraper',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'beautifulsoup4',
        'nltk',
        'keybert',
        'transformers',
        'tqdm',
        'spacy',
        'selenium',
        'aiohttp',
        'webdriver-manager',
        'python-dotenv'
        'flask'
        'axios'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts':[
            'brief_scraper=brief_scraper:main'
        ]
    }
)