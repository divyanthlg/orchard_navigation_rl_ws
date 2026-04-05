from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'orchard_data_collector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divyanthlg',
    maintainer_email='divyanthlg@todo.com',
    description='BC data collector for orchard navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collector = orchard_data_collector.data_collector_node:main',
        ],
    },
)
