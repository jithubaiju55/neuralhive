"""
Storage Manager — NeuralHive
=============================
Detects all available storage (USB, SSD, HDD, any drive).
Recommends best option. Remembers user's choice.
Works on Windows, Mac, Linux.
"""

import os
import sys
import json
import time
import shutil
import psutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum


class StorageType(Enum):
    NVME_SSD = "nvme_ssd"
    SATA_SSD = "sata_ssd"
    USB_3_2 = "usb_3.2"
    USB_3_0 = "usb_3.0"
    USB_2_0 = "usb_2.0"
    HDD = "hdd"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class StorageDevice:
    path: str
    label: str
    storage_type: StorageType
    total_gb: float
    free_gb: float
    read_speed_mbs: float       # estimated MB/s
    is_removable: bool
    is_recommended: bool = False
    warning: Optional[str] = None

    @property
    def speed_rating(self) -> str:
        if self.read_speed_mbs >= 3000:
            return "⚡ Ultra Fast"
        elif self.read_speed_mbs >= 500:
            return "✅ Fast"
        elif self.read_speed_mbs >= 200:
            return "⚠️  Acceptable"
        else:
            return "❌ Slow"

    @property
    def usable_for_models(self) -> bool:
        return self.read_speed_mbs >= 150 and self.free_gb >= 5


# Estimated read speeds by type (MB/s)
SPEED_ESTIMATES = {
    StorageType.NVME_SSD: 4000,
    StorageType.SATA_SSD: 500,
    StorageType.USB_3_2: 1000,
    StorageType.USB_3_0: 400,
    StorageType.USB_2_0: 40,
    StorageType.HDD: 120,
    StorageType.NETWORK: 50,
    StorageType.UNKNOWN: 100,
}


class StorageDetector:
    """
    Detects and ranks all available storage devices.
    Handles Windows drive letters, Linux/Mac mount points, USB detection.
    """

    CONFIG_FILE = Path.home() / ".neuralhive" / "storage_config.json"

    def __init__(self):
        self.devices: List[StorageDevice] = []
        self._saved_config: Optional[Dict] = self._load_config()

    def detect_all(self) -> List[StorageDevice]:
        """Detect all available storage devices."""
        self.devices = []

        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
            except (PermissionError, OSError):
                continue

            total_gb = usage.total / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)

            if total_gb < 1.0:  # Skip tiny partitions
                continue

            storage_type = self._detect_type(partition)
            speed = SPEED_ESTIMATES[storage_type]
            label = self._get_label(partition)
            is_removable = self._is_removable(partition)
            warning = self._get_warning(storage_type, free_gb)

            device = StorageDevice(
                path=partition.mountpoint,
                label=label,
                storage_type=storage_type,
                total_gb=round(total_gb, 1),
                free_gb=round(free_gb, 1),
                read_speed_mbs=speed,
                is_removable=is_removable,
                warning=warning,
            )
            self.devices.append(device)

        # Mark best recommendation
        self._mark_recommendation()
        return self.devices

    def _detect_type(self, partition) -> StorageType:
        """Detect storage type using Windows APIs first, speed test as fallback."""
        if sys.platform == "win32":
            detected = self._detect_type_windows(partition.mountpoint)
            if detected:
                return detected

        # Linux/Mac: check device name
        device = getattr(partition, 'device', '').lower()
        opts = partition.opts.lower() if partition.opts else ""
        if 'nvme' in device:
            return StorageType.NVME_SSD
        if 'sd' in device and 'usb' in opts:
            return StorageType.USB_3_0
        if 'sd' in device:
            return StorageType.SATA_SSD

        # Universal fallback: speed benchmark
        measured_speed = self._quick_speed_test(partition.mountpoint)
        if measured_speed > 2000:
            return StorageType.NVME_SSD
        elif measured_speed > 400:
            return StorageType.SATA_SSD
        elif measured_speed > 150:
            return StorageType.USB_3_0
        elif measured_speed > 30:
            return StorageType.HDD
        else:
            return StorageType.UNKNOWN

    def _detect_type_windows(self, mountpoint: str) -> Optional[StorageType]:
        """
        Windows-specific drive detection using multiple methods.
        Returns None if detection fails — falls back to speed test.
        """
        import subprocess
        drive_letter = mountpoint.rstrip("\\").rstrip("/").upper()
        if not drive_letter:
            return None

        # Method 1: PowerShell Get-PhysicalDisk — most accurate
        try:
            ps_cmd = (
                f"$d = Get-Partition | Where-Object {{$_.DriveLetter -eq '{drive_letter[0]}'}} | "
                f"Get-Disk | Get-PhysicalDisk; "
                f"Write-Output ($d.MediaType + '|' + $d.BusType)"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.strip().lower()
            if output and "|" in output:
                media_type, bus_type = output.split("|", 1)
                if "nvme" in bus_type or "nvme" in media_type:
                    return StorageType.NVME_SSD
                if "ssd" in media_type or "solid" in media_type:
                    return StorageType.SATA_SSD if "sata" in bus_type else StorageType.NVME_SSD
                if "usb" in bus_type:
                    return StorageType.USB_3_0
                if "hdd" in media_type or "unspecified" in media_type:
                    return StorageType.HDD
        except Exception:
            pass

        # Method 2: WMIC per-drive — fallback
        try:
            result = subprocess.run(
                ["wmic", "logicaldisk", "where",
                 f"DeviceID='{drive_letter[0]}:'",
                 "get", "DriveType"],
                capture_output=True, text=True, timeout=3
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip().isdigit()]
            if lines:
                drive_type = int(lines[0])
                # DriveType: 2=removable, 3=fixed, 4=network, 5=CDROM, 6=RAM
                if drive_type == 2:
                    return StorageType.USB_3_0   # removable — assume USB 3.0
                if drive_type == 3:
                    # Fixed drive — do a speed test to distinguish NVMe vs HDD
                    speed = self._quick_speed_test(mountpoint)
                    if speed > 1000:
                        return StorageType.NVME_SSD
                    elif speed > 300:
                        return StorageType.SATA_SSD
                    else:
                        return StorageType.HDD
        except Exception:
            pass

        return None

    def _quick_speed_test(self, path: str, size_mb: int = 10) -> float:
        """
        Quick sequential read speed test.
        Writes then reads a small test file.
        Returns MB/s estimate.
        """
        test_file = Path(path) / ".neuralhive_speedtest"
        try:
            # Write test
            data = os.urandom(size_mb * 1024 * 1024)
            with open(test_file, 'wb') as f:
                f.write(data)

            # Read test
            start = time.time()
            with open(test_file, 'rb') as f:
                _ = f.read()
            elapsed = time.time() - start

            speed = size_mb / elapsed if elapsed > 0 else 0
            return speed
        except (PermissionError, OSError):
            return 0
        finally:
            try:
                test_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _get_label(self, partition) -> str:
        """Human-readable label for drive."""
        if sys.platform == "win32":
            try:
                import ctypes
                label = ctypes.create_unicode_buffer(261)
                ctypes.windll.kernel32.GetVolumeInformationW(
                    partition.mountpoint, label, 261,
                    None, None, None, None, 0
                )
                vol_label = label.value.strip()
                if vol_label:
                    return f"{partition.mountpoint} ({vol_label})"
            except Exception:
                pass
            return partition.mountpoint
        else:
            return partition.mountpoint

    def _is_removable(self, partition) -> bool:
        """Check if drive is removable (USB etc)."""
        if sys.platform == "win32":
            try:
                import ctypes
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(
                    partition.mountpoint
                )
                # 2 = removable, 3 = fixed, 4 = network, 5 = CDROM
                return drive_type == 2
            except Exception:
                pass
        return False

    def _get_warning(self, storage_type: StorageType, free_gb: float) -> Optional[str]:
        warnings = []
        if storage_type == StorageType.USB_2_0:
            warnings.append("USB 2.0 detected — very slow for models. Prefer USB 3.0+ or SSD.")
        elif storage_type == StorageType.UNKNOWN:
            warnings.append("Drive speed unknown — if slow, model loading will be delayed.")
        if free_gb < 5:
            warnings.append(f"Only {free_gb:.1f}GB free — may not fit any model.")
        elif free_gb < 10:
            warnings.append(f"Only {free_gb:.1f}GB free — fits small models only (7B).")
        return "  ".join(warnings) if warnings else None

    def _mark_recommendation(self):
        """Mark the best storage option."""
        usable = [d for d in self.devices if d.usable_for_models]
        if not usable:
            return

        # Score: speed * 0.6 + free_space * 0.4 (normalized)
        max_speed = max(d.read_speed_mbs for d in usable)
        max_space = max(d.free_gb for d in usable)

        def score(d: StorageDevice) -> float:
            speed_score = d.read_speed_mbs / max_speed if max_speed > 0 else 0
            space_score = d.free_gb / max_space if max_space > 0 else 0
            return speed_score * 0.6 + space_score * 0.4

        best = max(usable, key=score)
        best.is_recommended = True

    def get_saved_path(self) -> Optional[str]:
        """Return previously saved storage path if still valid."""
        if not self._saved_config:
            return None
        path = self._saved_config.get("path")
        if path and Path(path).exists():
            return path
        return None

    def save_choice(self, path: str):
        """Save user's storage choice."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {"path": path, "set_at": time.time()}
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def _load_config(self) -> Optional[Dict]:
        """Load saved storage config."""
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def get_models_dir(self, base_path: str) -> Path:
        """Get the models directory within chosen storage."""
        models_dir = Path(base_path) / "neuralhive_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir