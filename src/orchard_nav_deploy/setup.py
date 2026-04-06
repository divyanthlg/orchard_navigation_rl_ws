from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'orchard_nav_deploy'

setup(
    name=package_name,
    version='0.3.0',
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
    description='BC policy deploy + DAgger for orchard navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = orchard_nav_deploy.policy_node:main',
            'dagger_supervisor = orchard_nav_deploy.dagger_supervisor_node:main',
            'cmd_vel_mux = orchard_nav_deploy.cmd_vel_mux_node:main',
            'dagger_status_display = orchard_nav_deploy.dagger_status_display:main',
        ],
    },
)
