#!/usr/bin/env python3
"""
Installer script for Aliyah ML Training Monitor
"""
import os
import sys
import platform
import subprocess
import shutil
import tempfile
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'

def print_step(message):
    print(f"{BLUE}▶ {message}{RESET}")

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def check_requirements():
    """Check if the system has the required tools."""
    print_step("Checking requirements")
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 6):
        print_error("Python 3.6+ is required")
        sys.exit(1)
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print_error("pip is not available")
        sys.exit(1)
    
    # Check if Rust/Cargo is installed
    cargo_installed = False
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.PIPE)
        cargo_installed = True
    except (subprocess.SubprocessError, FileNotFoundError):
        print_error("Rust/Cargo not found. Will try to install prebuilt binary.")
    
    return cargo_installed

def install_prebuilt_binary():
    """Download and install prebuilt binary for the platform."""
    print_step("Installing prebuilt binary")
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        binary_extension = ".exe"
        if machine == "amd64" or machine == "x86_64":
            arch = "x86_64-pc-windows-msvc"
        else:
            print_error(f"Unsupported Windows architecture: {machine}")
            return False
    elif system == "darwin":  # macOS
        binary_extension = ""
        if machine == "x86_64":
            arch = "x86_64-apple-darwin"
        elif machine == "arm64":
            arch = "aarch64-apple-darwin"
        else:
            print_error(f"Unsupported macOS architecture: {machine}")
            return False
    elif system == "linux":
        binary_extension = ""
        if machine == "x86_64":
            arch = "x86_64-unknown-linux-gnu"
        elif machine == "aarch64" or machine == "arm64":
            arch = "aarch64-unknown-linux-gnu"
        else:
            print_error(f"Unsupported Linux architecture: {machine}")
            return False
    else:
        print_error(f"Unsupported operating system: {system}")
        return False
    
    # GitHub release URL
    version = "0.1.0"  # Update as needed
    github_url = f"https://github.com/lovechants/Aliyah/releases/download/v{version}"
    
    if system == "windows":
        archive_name = f"aliyah-{arch}.zip"
        binary_name = f"aliyah{binary_extension}"
    else:
        archive_name = f"aliyah-{arch}.tar.gz"
        binary_name = "aliyah"
    
    download_url = f"{github_url}/{archive_name}"
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / archive_name
        
        # Download the archive
        try:
            import requests
            print_step(f"Downloading {download_url}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print_error(f"Failed to download binary: {e}")
            return False
        
        # Extract the archive
        try:
            if system == "windows":
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_path)
            else:
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(tmp_path)
        except Exception as e:
            print_error(f"Failed to extract archive: {e}")
            return False
        
        # Install the binary
        binary_path = tmp_path / binary_name
        if not binary_path.exists():
            print_error(f"Binary not found in archive: {binary_name}")
            return False
        
        # Determine installation directory
        if system == "windows":
            bin_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Aliyah" / "bin"
        else:
            bin_dir = Path.home() / ".local" / "bin"
        
        # Create directory if it doesn't exist
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy binary to installation directory
        dest_path = bin_dir / binary_name
        shutil.copy2(binary_path, dest_path)
        
        # Make binary executable on Unix-like systems
        if system != "windows":
            os.chmod(dest_path, 0o755)
        
        print_success(f"Installed binary to {dest_path}")
        
        # Add to PATH if not already in PATH
        if system != "windows":
            rc_files = []
            shell = os.environ.get("SHELL", "")
            
            if "bash" in shell:
                rc_files.append(Path.home() / ".bashrc")
            elif "zsh" in shell:
                rc_files.append(Path.home() / ".zshrc")
            
            for rc_file in rc_files:
                if rc_file.exists():
                    with open(rc_file, 'r') as f:
                        content = f.read()
                    
                    path_entry = f'export PATH="$PATH:{bin_dir}"'
                    if path_entry not in content:
                        with open(rc_file, 'a') as f:
                            f.write(f"\n# Added by Aliyah installer\n{path_entry}\n")
                        print_step(f"Added {bin_dir} to PATH in {rc_file}")
                        print_step(f"Please run 'source {rc_file}' or restart your terminal")
        
        return True

def install_python_package():
    """Install the Aliyah Python package."""
    print_step("Installing Python package")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "aliyah"], check=True)
        print_success("Installed Aliyah Python package")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install Python package: {e}")
        return False

def install_with_cargo():
    """Install Aliyah using Cargo."""
    print_step("Installing with Cargo...")
    
    try:
        subprocess.run(["cargo", "install", "aliyah"], check=True)
        print_success("Installed Aliyah using Cargo")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install with Cargo: {e}")
        return False

def main():
    """Main installation function."""
    print_step("Installing Aliyah - ML Training Monitor")
    
    cargo_installed = check_requirements()
    
    # Install the binary
    binary_installed = False
    if cargo_installed:
        binary_installed = install_with_cargo()
    
    if not binary_installed:
        binary_installed = install_prebuilt_binary()
    
    if not binary_installed:
        print_error("Failed to install Aliyah binary")
        sys.exit(1)
    
    # Install the Python package
    py_installed = install_python_package()
    
    if not py_installed:
        print_error("Failed to install Aliyah Python package")
        sys.exit(1)
    
    print_success("Installation complete!")
    print(f"{BLUE}▶ Usage: aliyah <script.py>{RESET}")

if __name__ == "__main__":
    main()
