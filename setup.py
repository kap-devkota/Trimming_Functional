from setuptools import setup, find_packages, Extension

module1 = Extension("libl3",
                   sources=["l3.c"])

setup(
    name='denoise',
    version='1',
    description='Denoising graphs using diffusion state distance link prediction',
    author='Henri Schmidt and Kapil Devkota',
    author_email='henri.schmidt@tufts.edu',
    url='https://github.com/kap-devkota/Trimming_Functional',
    packages=find_packages(exclude=('tests', 'docs', 'results', 'data')),
    ext_modules=[module1]
)
