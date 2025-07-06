# cleanup_trading_bot.py
"""
Script para limpiar y organizar el proyecto trading-bot
Ejecutar desde la raíz del proyecto
"""
import os
import shutil
import re
from pathlib import Path
from typing import List, Set

class ProjectCleaner:
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.files_to_delete = []
        self.suspicious_files = []
        self.unused_imports = {}
        
    def clean_cache_files(self):
        """Elimina archivos de cache y temporales"""
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/.pytest_cache",
            "**/.coverage",
            "**/*.log",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.bak",
            "**/*~",
            "**/*.swp"
        ]
        
        print("🧹 Limpiando archivos de cache...")
        for pattern in cache_patterns:
            for file_path in self.root.glob(pattern):
                if file_path.is_file():
                    self.files_to_delete.append(file_path)
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"  ✓ Eliminado directorio: {file_path}")
    
    def find_suspicious_files(self):
        """Encuentra archivos que podrían contener información sensible"""
        suspicious_patterns = [
            r"api_key|API_KEY",
            r"password|PASSWORD", 
            r"secret|SECRET",
            r"token|TOKEN",
            r"private_key|PRIVATE_KEY"
        ]
        
        print("\n🔍 Buscando archivos con posible información sensible...")
        
        for py_file in self.root.glob("**/*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in suspicious_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Verificar si es un string hardcodeado
                        if re.search(f'{pattern}\\s*=\\s*["\']\\w+["\']', content):
                            self.suspicious_files.append((py_file, pattern))
                            break
            except Exception as e:
                print(f"  ⚠️  Error leyendo {py_file}: {e}")
    
    def find_unused_imports(self):
        """Encuentra imports no utilizados"""
        print("\n🔍 Analizando imports no utilizados...")
        
        for py_file in self.root.glob("**/*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Encontrar todos los imports
                import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
                imports = re.findall(import_pattern, content)
                
                unused = []
                for module, items in imports:
                    # Simplificado - en producción usar AST
                    imported_names = [item.strip() for item in items.split(',')]
                    for name in imported_names:
                        name = name.split(' as ')[0].strip()
                        # Verificar si se usa en el código
                        if not re.search(f'\\b{name}\\b', content.replace(f'import {name}', '')):
                            unused.append(name)
                
                if unused:
                    self.unused_imports[py_file] = unused
                    
            except Exception as e:
                print(f"  ⚠️  Error analizando {py_file}: {e}")
    
    def suggest_structure_improvements(self):
        """Sugiere mejoras en la estructura del proyecto"""
        print("\n📁 Analizando estructura del proyecto...")
        
        suggestions = []
        
        # Verificar si existe .gitignore
        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.exists():
            suggestions.append("Crear archivo .gitignore")
            self._create_gitignore()
        
        # Verificar estructura de carpetas
        recommended_dirs = [
            "src/strategies",
            "src/indicators", 
            "src/backtesting",
            "src/risk_management",
            "src/execution",
            "src/data",
            "src/monitoring",
            "src/utils",
            "tests",
            "docs",
            "config",
            "scripts"
        ]
        
        for dir_path in recommended_dirs:
            full_path = self.root / dir_path
            if not full_path.exists():
                suggestions.append(f"Crear directorio: {dir_path}")
        
        # Verificar archivos importantes
        important_files = [
            "requirements.txt",
            "README.md",
            ".env.example",
            "setup.py"
        ]
        
        for file_name in important_files:
            if not (self.root / file_name).exists():
                suggestions.append(f"Crear archivo: {file_name}")
        
        return suggestions
    
    def _create_gitignore(self):
        """Crea un .gitignore apropiado para el proyecto"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
logs/
data/cache/
backtest_results/
*.db
*.sqlite

# Sensitive
.env
.env.*
config/secrets.yaml
config/api_keys.json
*.pem
*.key

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Package files
*.egg-info/
dist/
build/
"""
        
        with open(self.root / ".gitignore", "w") as f:
            f.write(gitignore_content)
        print("  ✓ Creado archivo .gitignore")
    
    def create_env_example(self):
        """Crea un archivo .env.example"""
        env_example = """# Exchange API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Configuration
TRADING_MODE=paper  # paper or live
MAX_POSITION_SIZE=1000  # in USD
RISK_PER_TRADE=0.02  # 2%

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
"""
        
        with open(self.root / ".env.example", "w") as f:
            f.write(env_example)
        print("  ✓ Creado archivo .env.example")
    
    def generate_report(self):
        """Genera reporte de limpieza"""
        print("\n" + "="*50)
        print("📊 REPORTE DE LIMPIEZA")
        print("="*50)
        
        if self.files_to_delete:
            print(f"\n🗑️  Archivos para eliminar ({len(self.files_to_delete)}):")
            for file in self.files_to_delete[:10]:  # Mostrar máximo 10
                print(f"  - {file}")
            if len(self.files_to_delete) > 10:
                print(f"  ... y {len(self.files_to_delete) - 10} más")
        
        if self.suspicious_files:
            print(f"\n⚠️  Archivos con posible información sensible ({len(self.suspicious_files)}):")
            for file, pattern in self.suspicious_files:
                print(f"  - {file} (contiene: {pattern})")
        
        if self.unused_imports:
            print(f"\n📦 Archivos con imports no utilizados ({len(self.unused_imports)}):")
            for file, imports in list(self.unused_imports.items())[:5]:
                print(f"  - {file}: {', '.join(imports[:3])}")
        
        suggestions = self.suggest_structure_improvements()
        if suggestions:
            print(f"\n💡 Sugerencias de mejora ({len(suggestions)}):")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
    
    def execute_cleanup(self):
        """Ejecuta la limpieza con confirmación"""
        if self.files_to_delete:
            response = input(f"\n¿Eliminar {len(self.files_to_delete)} archivos? (s/n): ")
            if response.lower() == 's':
                for file in self.files_to_delete:
                    try:
                        file.unlink()
                        print(f"  ✓ Eliminado: {file}")
                    except Exception as e:
                        print(f"  ✗ Error eliminando {file}: {e}")
        
        # Crear archivos sugeridos
        if not (self.root / ".env.example").exists():
            response = input("\n¿Crear archivo .env.example? (s/n): ")
            if response.lower() == 's':
                self.create_env_example()

def main():
    print("🚀 Trading Bot Project Cleaner")
    print("="*50)
    
    cleaner = ProjectCleaner()
    
    # Ejecutar análisis
    cleaner.clean_cache_files()
    cleaner.find_suspicious_files()
    cleaner.find_unused_imports()
    
    # Generar reporte
    cleaner.generate_report()
    
    # Ejecutar limpieza
    cleaner.execute_cleanup()
    
    print("\n✅ Limpieza completada!")

if __name__ == "__main__":
    main()