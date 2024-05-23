from setuptools import setup

package_name = 'mpc_comparison'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='factslabegmc',
    maintainer_email='evannsmcuadrado@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'holybroMPC = mpc_comparison.holybroMPC:main',
            # 'testImportMPC = mpc_comparison.testImportMPC:main',
            'christianBased_MPC = mpc_comparison.christianbased_holybroMPC:main',
            'holybroMPC = mpc_comparison.holybroMPC:main',
        ],
    },
)
