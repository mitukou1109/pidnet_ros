from setuptools import find_packages, setup

package_name = "pidnet_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mitsuhiro Sakamoto",
    maintainer_email="mitukou1109@gmail.com",
    description="ROS 2 wrapper for PIDNet",
    license="BSD",
    entry_points={
        "console_scripts": [
            "pidnet_ros_node = pidnet_ros.pidnet_ros:main",
        ],
    },
)