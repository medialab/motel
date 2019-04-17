from setuptools import setup, find_packages

with open('./README.md', 'r') as f:
    long_description = f.read()

setup(name='motel',
      version='0.1.0',
      description='Exploration tool for word embeddings.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/medialab/motel',
      license='MIT',
      author='Guillaume Plique',
      keywords='nlp',
      python_requires='>=3',
      packages=find_packages(exclude=['ftest', 'test']),
      install_requires=[
        'gensim>=3',
        'spacy>=2.1.3',
        'tqdm>=4.31.1'
      ],
      entry_points={
        'console_scripts': ['motel=motel.cli.__main__:main']
      },
      zip_safe=True)
