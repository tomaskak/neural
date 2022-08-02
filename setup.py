import setuptools

setuptools.setup(
    name='neural',
    version='0.0.0',
    author='Tomas Kaknevicius',
    author_email='tkaknevicius@gmail.com',
    description='AI utilities',
    url='https://github.com/tomaskak/neural.git',
    licence='MIT',
    packages=['neural'],
    install_requires=['numpy', 'pytorch', 'gym', 'pyglet', 'gym[mujoco]', 'plotly', 'pandas', 'dash'])