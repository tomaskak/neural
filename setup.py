import setuptools

setuptools.setup(
    name="neural",
    version="0.0.0",
    author="Tomas Kaknevicius",
    author_email="tkaknevicius@gmail.com",
    description="AI utilities",
    url="https://github.com/tomaskak/neural.git",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "gym",
        "pyglet",
        "gym[mujoco]",
        "plotly",
        "pandas",
        "dash",
    ],
)
