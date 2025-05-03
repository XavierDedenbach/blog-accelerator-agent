#!/usr/bin/env python
"""
Setup script for the enhanced research components of the Blog Accelerator Agent.

This script:
1. Creates required directories for research data
2. Creates default configuration files if needed
3. Installs Python dependencies if not already installed
4. Checks API keys and provides guidance for missing configurations
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SetupError(Exception):
    """Exception raised for errors in setup process."""
    pass


def create_directories() -> List[str]:
    """
    Create required directories for research data.
    
    Returns:
        List of directories created
    """
    directories = [
        "data/blacklist",
        "data/firecrawl_cache",
        "data/validation_cache",
        "data/visual_assets",
        "data/research_components",
        "data/analogies",
        "data/tracker_yaml"
    ]
    
    created = []
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            created.append(str(path))
            logger.info(f"Created directory: {path}")
    
    return created


def create_default_files() -> List[str]:
    """
    Create default configuration files if they don't exist.
    
    Returns:
        List of files created
    """
    created = []
    
    # Default blacklist
    blacklist_path = Path("data/blacklist/default.json")
    if not blacklist_path.exists():
        default_blacklist = [
            "example.com",
            "untrusted-source.com",
            "fake-news.org",
            "conspiracy-theories.net"
        ]
        blacklist_path.write_text(json.dumps(default_blacklist, indent=2))
        created.append(str(blacklist_path))
        logger.info(f"Created default blacklist: {blacklist_path}")
    
    # Example .env file if not exists
    env_path = Path(".env")
    if not env_path.exists():
        env_content = """# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Groq
GROQ_API_KEY=grq-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# MongoDB
MONGODB_URI=mongodb://localhost:27017

# Brave Search (Premium Tier)
BRAVE_API_KEY=brv-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Firecrawl MCP
FIRECRAWL_SERVER=http://firecrawl:4000

# Opik MCP
OPIK_SERVER=http://opik:7000
"""
        env_path.write_text(env_content)
        created.append(str(env_path))
        logger.info(f"Created example .env file: {env_path}")
    
    return created


def check_python_dependencies() -> Dict[str, bool]:
    """
    Check if required Python dependencies are installed.
    
    Returns:
        Dictionary mapping package names to installation status
    """
    required_packages = [
        "pymongo",
        "langchain",
        "openai",
        "requests",
        "python-dotenv",
        "pytest",
        "fastapi",
        "uvicorn",
        "pyyaml"
    ]
    
    package_status = {}
    for package in required_packages:
        try:
            __import__(package)
            package_status[package] = True
            logger.info(f"Found package: {package}")
        except ImportError:
            package_status[package] = False
            logger.warning(f"Missing package: {package}")
    
    return package_status


def install_missing_dependencies(package_status: Dict[str, bool]) -> List[str]:
    """
    Install missing Python dependencies.
    
    Args:
        package_status: Dictionary mapping package names to installation status
        
    Returns:
        List of packages installed
    """
    missing_packages = [pkg for pkg, installed in package_status.items() if not installed]
    
    if not missing_packages:
        logger.info("All required packages are already installed")
        return []
    
    installed = []
    for package in missing_packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            installed.append(package)
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
    
    return installed


def check_api_keys() -> Dict[str, bool]:
    """
    Check if required API keys are set in environment variables.
    
    Returns:
        Dictionary mapping API keys to availability status
    """
    # Try to load from .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        try:
            import dotenv
            dotenv.load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env loading")
    
    required_keys = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "MONGODB_URI",
        "BRAVE_API_KEY",
        "FIRECRAWL_SERVER",
        "OPIK_SERVER"
    ]
    
    key_status = {}
    for key in required_keys:
        value = os.environ.get(key)
        key_status[key] = bool(value and not value.startswith(("xx", "sk-xx", "grq-xx", "brv-xx")))
        
        if key_status[key]:
            logger.info(f"Found API key: {key}")
        else:
            logger.warning(f"Missing or example API key: {key}")
    
    return key_status


def main():
    """Main function to run the setup process."""
    parser = argparse.ArgumentParser(description='Setup the Blog Accelerator research components')
    parser.add_argument('--install-deps', action='store_true', help='Install missing dependencies')
    args = parser.parse_args()
    
    try:
        # Create directories
        logger.info("Creating required directories...")
        created_dirs = create_directories()
        
        # Create default files
        logger.info("Creating default configuration files...")
        created_files = create_default_files()
        
        # Check Python dependencies
        logger.info("Checking Python dependencies...")
        package_status = check_python_dependencies()
        
        # Install missing dependencies if requested
        installed_packages = []
        if args.install_deps:
            logger.info("Installing missing dependencies...")
            installed_packages = install_missing_dependencies(package_status)
        
        # Check API keys
        logger.info("Checking API keys...")
        key_status = check_api_keys()
        
        # Print summary
        print("\n=== Setup Summary ===")
        if created_dirs:
            print(f"Created {len(created_dirs)} directories")
        else:
            print("No new directories created")
            
        if created_files:
            print(f"Created {len(created_files)} default files")
        else:
            print("No new files created")
            
        missing_packages = [pkg for pkg, installed in package_status.items() if not installed]
        if missing_packages:
            if installed_packages:
                print(f"Installed {len(installed_packages)} packages: {', '.join(installed_packages)}")
            else:
                print(f"Missing {len(missing_packages)} packages: {', '.join(missing_packages)}")
                print("Run with --install-deps to install missing packages")
        else:
            print("All required packages are installed")
            
        missing_keys = [key for key, available in key_status.items() if not available]
        if missing_keys:
            print(f"Missing {len(missing_keys)} API keys: {', '.join(missing_keys)}")
            print("Edit your .env file to add the missing keys")
        else:
            print("All required API keys are available")
        
        if not missing_packages and not missing_keys:
            print("\nSetup complete! Your research environment is ready.")
        else:
            print("\nSetup incomplete. Please address the issues above.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 