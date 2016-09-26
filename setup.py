from setuptools import setup


PACKAGES = [
    'mda',
    'mda.tests'
]

def setup_package():
    setup(
        name="MarginalizedDenoisingAutoEncoder",
        version='0.1.0',
        description='Python Package for the Marginalized Denoising Autoencoder Algorithm',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/MarginalizedDenoisingAutoencoder',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
