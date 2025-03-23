from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
        name="aliyah",
        version="0.1.0",
        packages=find_packages(),
        description="ML Training Visualization Tool Hooks",
        long_description=long_description, 
        long_description_content_type="text/markdown",
        author="j",
        author_email="j@07-i.co | n50513186@gmail.com",
        url="https://github.com/lovechants/Aliyah",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Monitoring",
        ],
        python_requires=">=3.6",
        install_requires=[
            "pyzmq>=22.0.0",
        ],
        entry_points={
            "console_scripts": [
                "aliyah-monitor=aliyah:trainingmonitor",
            ],
        },
    )

