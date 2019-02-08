from setuptools import setup, find_packages

long_description = open('README.md').read()
VERSION = '0.0.1'
setup(name='ovotools',
      version=VERSION,
      author="Ilya Ovodov",
      author_email="iovodov@gmail.com",
      url="https://github.com/IlyaOvodov/OvoTools",
      #download_url='https://github.com/bstriner/keras-tqdm/tarball/v{}'.format(VERSION),
      description="Useful stuff for DL",
      long_description=long_description,
      #keywords=['keras', 'tqdm', 'progress', 'progressbar', 'ipython', 'jupyter'],
      license='MIT',
      classifiers=[
          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',
          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'
      ],
      packages=find_packages())
