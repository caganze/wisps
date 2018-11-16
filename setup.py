import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(name='wisps',
      version='1.0',
      description='python code for my research',
      author='Christian Aganze',
      author_email='caganze@gmail.com',
      url='',
      packages=['wisps', 'wisps.data_analysis'],
     )