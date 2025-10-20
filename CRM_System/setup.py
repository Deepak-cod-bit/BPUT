"""
Setup script for AI Calling System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-calling-system",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="AI-Enabled CRM System with Automated AI Calling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-calling-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Communications :: Telephony",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "advanced": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "spacy>=3.6.0",
        ],
        "api": [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "gunicorn>=20.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-calling=integration_layer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_calling_system": [
            "*.json",
            "*.yaml",
            "*.yml",
        ],
    },
)
