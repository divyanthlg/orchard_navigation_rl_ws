from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'orchard_bc_training'

setup(
    name=package_name,
    version='0.7.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divyanthlg',
    maintainer_email='divyanthlg@todo.com',
    description='BC training + deploy for orchard navigation (VAE + GRU + MLP)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Training utilities
            'build_cache      = orchard_bc_training.build_cache:main',
            'train            = orchard_bc_training.train:main',
            # ROS nodes
            'bc_data_collector = orchard_bc_training.bc_data_collector_node:main',
            'bc_policy_node    = orchard_bc_training.bc_policy_node:main',
            'bc_cmd_vel_mux    = orchard_bc_training.bc_cmd_vel_mux_node:main',
            'bc_status_display = orchard_bc_training.bc_status_display:main',
            'bc_viz_node       = orchard_bc_training.bc_viz_node:main',
        ],
    },
)
