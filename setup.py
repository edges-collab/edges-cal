from setuptools import setup
import os, sys
import os.path as op
import json


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


setup_args = {
    'name': 'calibrate',
    'author': 'EDGES Team',
    'url': 'https://github.com/edges-collab/cal_coefficients',
    'license': 'BSD',
    'description': 'Calibrate EDGES spectra',
    'package_dir': {'calibrate': 'calibrate'},
    'packages': ['calibrate'],
    'include_package_data': True,
    'version': '0.1.0',
    'install_requires': [
        'numpy',
        'matplotlib',
        'scipy'
    ],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
