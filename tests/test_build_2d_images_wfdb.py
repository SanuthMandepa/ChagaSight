# tests/test_build_2d_images_wfdb.py
"""
Test suite for build_2d_images_wfdb.py: Checks all required files, folders, code structure,
and data presence before running the image generation script.

Run this test file with:
python -m unittest tests.test_build_2d_images_wfdb
"""

import unittest
import os
import sys
from pathlib import Path
import importlib.util

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

class TestBuild2DImagesWFDB(unittest.TestCase):
    """Test suite for prerequisites of build_2d_images_wfdb.py"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.official_wfdb_dir = self.project_root / "data" / "official_wfdb"
        self.processed_img_dir = self.project_root / "data" / "processed" / "2d_images"
        self.src_dir = self.project_root / "src" / "preprocessing"
        self.scripts_dir = self.project_root / "scripts"
        self.test_dir = self.project_root / "tests"

    def test_project_structure(self):
        """Check main project folders exist"""
        self.assertTrue(self.project_root.exists(), "Project root folder missing")
        self.assertTrue(self.official_wfdb_dir.exists(), "data/official_wfdb missing")
        self.assertTrue(self.src_dir.exists(), "src/preprocessing missing")
        self.assertTrue(self.scripts_dir.exists(), "scripts folder missing")

    def test_wfdb_data_folders(self):
        """Check WFDB folders for each dataset exist and have files"""
        datasets = ["ptbxl", "sami_trop", "code15"]
        for ds in datasets:
            ds_path = self.official_wfdb_dir / ds
            self.assertTrue(ds_path.exists(), f"{ds} WFDB folder missing")
            hea_files = list(ds_path.rglob("*.hea"))
            self.assertTrue(len(hea_files) > 0, f"No .hea files in {ds} folder")
            dat_files = list(ds_path.rglob("*.dat"))
            self.assertTrue(len(dat_files) > 0, f"No .dat files in {ds} folder")

    def test_preprocessing_modules_exist(self):
        """Check required preprocessing modules exist and can be imported"""
        modules = [
            "baseline_removal.py",
            "resample.py",
            "normalization.py",
            "image_embedding.py"
        ]
        for mod in modules:
            mod_path = self.src_dir / mod
            self.assertTrue(mod_path.exists(), f"{mod} missing in src/preprocessing")
            
            # Test import
            spec = importlib.util.spec_from_file_location(mod[:-3], str(mod_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(module, f"Failed to import {mod}")

    def test_build_script_exists(self):
        """Check build_2d_images_wfdb.py exists and is importable"""
        script_path = self.scripts_dir / "build_2d_images_wfdb.py"
        self.assertTrue(script_path.exists(), "build_2d_images_wfdb.py missing")
        
        # Test basic import (syntax check)
        spec = importlib.util.spec_from_file_location("build_2d_images_wfdb", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(module, "Script syntax/import error")

    def test_output_folder_structure(self):
        """Check output folder is ready (created by script, but test parent exists)"""
        self.assertTrue(self.processed_img_dir.parent.exists(), "data/processed missing")
        # Script creates subfolders automatically, so no need to pre-create

    def test_requirements_txt_exists(self):
        """Check requirements.txt exists for dependencies"""
        req_path = self.project_root / "requirements.txt"
        self.assertTrue(req_path.exists(), "requirements.txt missing")
        with open(req_path) as f:
            content = f.read()
            self.assertIn("wfdb", content, "wfdb missing from requirements")
            self.assertIn("numpy", content, "numpy missing from requirements")
            self.assertIn("tqdm", content, "tqdm missing from requirements")

    def test_venv_activation(self):
        """Basic check that venv is active (optional sanity)"""
        self.assertIn("venv", sys.executable.lower(), "Run from activated venv")

if __name__ == "__main__":
    unittest.main()