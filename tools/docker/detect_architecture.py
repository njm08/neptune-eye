
import platform
import os
from enum import Enum

class Architecture(Enum):
    ARM64 = 'arm64'
    JETPACK6 = 'jetpack6'
    X86_64 = 'x86_64'
    UNKNOWN = 'unknown'

def detect_architecture() -> Architecture:
    """
    Detect the underlying architecture.
    
    Returns:
        Architecture: Architecture type
    """
    machine = platform.machine().lower()
    architecture = Architecture.UNKNOWN
    # Check if running on Jetson (presence of jetson-specific files)
    if os.path.exists('/etc/nv_tegra_release'):
        # Parse JetPack version from the release file
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                # JetPack 6.x detection
                if 'R36' in content or 'R37' in content:
                    return 'jetpack6'
        except:
            pass
        architecture = Architecture.JETPACK6
    
    # Check for ARM64 architecture
    if machine in ['arm64', 'aarch64']:
        architecture = Architecture.ARM64

    # Check for x86_64/AMD64
    if machine in ['x86_64', 'amd64']:
        architecture = Architecture.X86_64
    
    print(f"Detected architecture: {architecture.value}")
    
    return architecture
