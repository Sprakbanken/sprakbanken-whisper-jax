from setuptools import setup, find_packages

setup(
    name='sprakbanken_whisper_jax',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sprakbanken-whisper-jax=sprakbanken_whisper_jax.run_whisper_jax:main',
        ],
    },
    install_requires=[
        'whisper-jax @ git+https://github.com/sanchit-gandhi/whisper-jax.git#egg=whisper-jax',
        'jax',
    ],
)