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
            'ros2_acados = mpc_comparison.ros2_acados_works:main',
            'ros2_pxy_load = mpc_comparison.ros2_pxy_load_works:main',
            'ros2_px4_integration = mpc_comparison.ros2_px4_integration:main',
            'ros2_traj_work = mpc_comparison.trajectory_work:main',
            'MPC_done = mpc_comparison.working_MPC:main',
        ],
    },
)
