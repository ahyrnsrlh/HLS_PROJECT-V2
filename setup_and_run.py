#!/usr/bin/env python3
"""
Script setup otomatis untuk sistem klasifikasi SDG
Menjalankan semua langkah setup dan training dalam satu command
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class SDGSetupRunner:
    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.venv_dir = self.project_dir / "sdg_env"
        self.data_file = self.project_dir / "data" / "2503_to_3336_preprocessing_labeling.csv"
        
    def print_step(self, step, message):
        """Print formatted step message"""
        print(f"\n{'='*60}")
        print(f"LANGKAH {step}: {message}")
        print('='*60)
        
    def run_command(self, command, shell=True):
        """Run command with error handling"""
        try:
            print(f"Menjalankan: {command}")
            result = subprocess.run(command, shell=shell, check=True, 
                                  capture_output=True, text=True)
            print("✅ Berhasil!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            print(f"Output: {e.output}")
            return False
            
    def check_python(self):
        """Check Python version"""
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            version = result.stdout.strip()
            print(f"Python ditemukan: {version}")
            
            # Check if Python >= 3.8
            version_num = version.split()[1]
            major, minor = map(int, version_num.split(".")[:2])
            if major < 3 or (major == 3 and minor < 8):
                print("❌ Python 3.8+ diperlukan!")
                return False
            return True
        except Exception as e:
            print(f"❌ Error checking Python: {e}")
            return False
            
    def check_data(self):
        """Check if data file exists"""
        if not self.data_file.exists():
            print(f"❌ File data tidak ditemukan: {self.data_file}")
            print("Pastikan file data ada di lokasi yang benar!")
            return False
        print(f"✅ File data ditemukan: {self.data_file}")
        return True
        
    def setup_venv(self):
        """Setup virtual environment"""
        if self.venv_dir.exists():
            print("Virtual environment sudah ada, skip...")
            return True
            
        print("Membuat virtual environment...")
        cmd = f'"{sys.executable}" -m venv "{self.venv_dir}"'
        return self.run_command(cmd)
        
    def get_venv_python(self):
        """Get path to virtual environment Python"""
        if os.name == 'nt':  # Windows
            return self.venv_dir / "Scripts" / "python.exe"
        else:  # Unix-like
            return self.venv_dir / "bin" / "python"
            
    def install_requirements(self):
        """Install requirements in virtual environment"""
        venv_python = self.get_venv_python()
        requirements_file = self.project_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ File requirements.txt tidak ditemukan!")
            return False
            
        cmd = f'"{venv_python}" -m pip install -r "{requirements_file}"'
        return self.run_command(cmd)
        
    def run_training(self):
        """Run model training"""
        venv_python = self.get_venv_python()
        train_script = self.project_dir / "train_model.py"
        
        if not train_script.exists():
            print("❌ Script training tidak ditemukan!")
            return False
            
        print("Mulai training model... (ini mungkin memakan waktu beberapa menit)")
        cmd = f'"{venv_python}" "{train_script}"'
        return self.run_command(cmd)
        
    def run_prediction_demo(self):
        """Run prediction demo"""
        venv_python = self.get_venv_python()
        predict_script = self.project_dir / "predict_sample.py"
        
        if not predict_script.exists():
            print("❌ Script prediksi tidak ditemukan!")
            return False
            
        cmd = f'"{venv_python}" "{predict_script}"'
        return self.run_command(cmd)
        
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 MEMULAI SETUP SISTEM KLASIFIKASI SDG")
        print(f"Direktori proyek: {self.project_dir}")
        
        # Step 1: Check prerequisites
        self.print_step(1, "MEMERIKSA PRASYARAT")
        if not self.check_python():
            return False
        if not self.check_data():
            return False
            
        # Step 2: Setup virtual environment
        self.print_step(2, "SETUP VIRTUAL ENVIRONMENT")
        if not self.setup_venv():
            return False
            
        # Step 3: Install dependencies
        self.print_step(3, "INSTALL DEPENDENCIES")
        if not self.install_requirements():
            return False
            
        # Step 4: Run training
        self.print_step(4, "TRAINING MODEL")
        if not self.run_training():
            return False
            
        # Step 5: Demo prediction
        self.print_step(5, "DEMO PREDIKSI")
        if not self.run_prediction_demo():
            return False
            
        # Success message
        self.print_step("✅", "SETUP SELESAI!")
        print("\n🎉 Sistem klasifikasi SDG berhasil disetup dan dijalankan!")
        print(f"\nUntuk aktivasi manual virtual environment:")
        if os.name == 'nt':
            print(f'"{self.venv_dir}\\Scripts\\Activate.ps1"')
        else:
            print(f'source "{self.venv_dir}/bin/activate"')
            
        print(f"\nModel tersimpan di: {self.project_dir / 'models'}")
        print(f"Hasil visualisasi di: {self.project_dir / 'outputs'}")
        
        return True

def main():
    """Main function"""
    setup_runner = SDGSetupRunner()
    
    try:
        success = setup_runner.run_setup()
        if success:
            print("\n✅ SETUP BERHASIL SEMPURNA!")
            sys.exit(0)
        else:
            print("\n❌ SETUP GAGAL!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error tidak terduga: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
