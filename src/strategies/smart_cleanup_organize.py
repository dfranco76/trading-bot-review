# smart_cleanup_organize.py
"""
Script inteligente para limpiar y organizar el proyecto trading-bot
Ignora carpetas de entorno virtual y se enfoca en el código del proyecto
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import ast
import re

class SmartProjectOrganizer:
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.venv_folders = {'venv', 'env', '.venv', 'virtualenv'}
        self.project_files = []
        self.organization_plan = {}
        
    def analyze_project(self):
        """Analiza solo los archivos del proyecto (ignora venv)"""
        print("🔍 Analizando estructura del proyecto...\n")
        
        # Encontrar archivos Python del proyecto
        for py_file in self.root.glob("**/*.py"):
            # Ignorar archivos en carpetas de entorno virtual
            if any(venv in py_file.parts for venv in self.venv_folders):
                continue
            if "__pycache__" in str(py_file):
                continue
                
            self.project_files.append(py_file)
        
        print(f"📁 Archivos Python encontrados: {len(self.project_files)}")
        
        # Analizar y categorizar archivos
        self._categorize_files()
        
    def _categorize_files(self):
        """Categoriza archivos por su contenido"""
        categories = {
            'strategies': [],
            'indicators': [],
            'backtesting': [],
            'risk_management': [],
            'execution': [],
            'data': [],
            'monitoring': [],
            'utils': [],
            'main': [],
            'tests': []
        }
        
        for file in self.project_files:
            content = file.read_text(encoding='utf-8', errors='ignore').lower()
            filename = file.name.lower()
            
            # Categorizar por nombre o contenido
            if 'strategy' in filename or 'class.*strategy' in content:
                categories['strategies'].append(file)
            elif 'indicator' in filename or 'sma|ema|rsi|macd' in content:
                categories['indicators'].append(file)
            elif 'backtest' in filename or 'backtest' in content:
                categories['backtesting'].append(file)
            elif 'risk' in filename or 'risk_management|stop_loss|position_size' in content:
                categories['risk_management'].append(file)
            elif 'order' in filename or 'execution' in filename or 'execute_order' in content:
                categories['execution'].append(file)
            elif 'data' in filename or 'fetch_data|download_data' in content:
                categories['data'].append(file)
            elif 'monitor' in filename or 'alert|notification' in content:
                categories['monitoring'].append(file)
            elif 'test_' in filename or 'tests' in str(file.parent):
                categories['tests'].append(file)
            elif filename in ['main.py', 'app.py', 'run.py', 'bot.py']:
                categories['main'].append(file)
            else:
                categories['utils'].append(file)
        
        self.organization_plan = categories
        
    def show_organization_plan(self):
        """Muestra el plan de organización propuesto"""
        print("\n📋 Plan de Organización Propuesto:")
        print("="*50)
        
        for category, files in self.organization_plan.items():
            if files:
                print(f"\n📁 src/{category}/")
                for file in files:
                    print(f"  └── {file.name}")
    
    def clean_project(self):
        """Limpia archivos innecesarios"""
        print("\n🧹 Limpiando proyecto...")
        
        # Limpiar archivos de cache
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/*.log",
            "**/.coverage"
        ]
        
        deleted = 0
        for pattern in cache_patterns:
            for path in self.root.glob(pattern):
                # Ignorar si está en venv
                if any(venv in path.parts for venv in self.venv_folders):
                    continue
                    
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    deleted += 1
                    print(f"  ✓ Eliminado: {path}")
                except Exception as e:
                    print(f"  ✗ Error eliminando {path}: {e}")
        
        print(f"\n✅ Archivos eliminados: {deleted}")
    
    def check_imports(self):
        """Verifica imports no utilizados en archivos del proyecto"""
        print("\n📦 Verificando imports no utilizados...")
        
        unused_imports = {}
        
        for file in self.project_files[:10]:  # Solo primeros 10 para no saturar
            try:
                content = file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                # Obtener todos los imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        imports.extend([alias.name for alias in node.names])
                
                # Verificar uso (simplificado)
                unused = []
                for imp in imports:
                    if content.count(imp) == 1:  # Solo aparece en el import
                        unused.append(imp)
                
                if unused:
                    unused_imports[file.name] = unused
                    
            except:
                continue
        
        if unused_imports:
            print("\nArchivos con imports posiblemente no utilizados:")
            for file, imports in list(unused_imports.items())[:5]:
                print(f"  - {file}: {', '.join(imports[:3])}")
    
    def create_project_structure(self):
        """Crea la estructura de carpetas recomendada"""
        print("\n🏗️ Creando estructura de proyecto...")
        
        directories = [
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
            "scripts",
            "logs"
        ]
        
        for dir_path in directories:
            full_path = self.root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True)
                print(f"  ✓ Creado: {dir_path}/")
                
                # Crear __init__.py en carpetas de Python
                if dir_path.startswith("src/"):
                    init_file = full_path / "__init__.py"
                    init_file.write_text("", encoding='utf-8')
    
    def create_readme(self):
        """Crea un README.md profesional"""
        readme_content = """# Trading Bot

Sistema automatizado de trading con múltiples estrategias y gestión de riesgo avanzada.

## 🚀 Características

- **Múltiples Estrategias**: Momentum, Mean Reversion, Arbitraje
- **Gestión de Riesgo**: Stop-loss dinámico, sizing inteligente
- **Backtesting**: Motor completo con métricas detalladas
- **Ejecución**: Soporte para múltiples exchanges
- **Monitoreo**: Dashboard en tiempo real y alertas

## 📁 Estructura del Proyecto

```
trading-bot/
├── src/
│   ├── strategies/      # Estrategias de trading
│   ├── indicators/      # Indicadores técnicos
│   ├── backtesting/     # Motor de backtesting
│   ├── risk_management/ # Gestión de riesgo
│   ├── execution/       # Ejecución de órdenes
│   ├── data/           # Manejo de datos
│   ├── monitoring/     # Sistema de monitoreo
│   └── utils/          # Utilidades
├── tests/              # Tests unitarios
├── config/             # Configuración
├── docs/               # Documentación
└── scripts/            # Scripts de utilidad
```

## 🛠️ Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/trading-bot.git
cd trading-bot
```

2. Crear entorno virtual:
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus API keys
```

## 🚦 Uso

### Modo Paper Trading
```bash
python main.py --mode paper
```

### Backtesting
```bash
python -m src.backtesting.run --strategy momentum --period 30d
```

### Trading en Vivo
```bash
python main.py --mode live --strategy mean_reversion
```

## ⚙️ Configuración

Ver `config/config.yaml` para opciones de configuración.

## 🧪 Tests

```bash
pytest tests/
```

## 📊 Métricas de Performance

- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## ⚠️ Disclaimer

Este bot es para fines educativos. El trading conlleva riesgos significativos.

## 📝 Licencia

MIT License
"""
        
        readme_path = self.root / "README.md"
        if not readme_path.exists():
            readme_path.write_text(readme_content, encoding='utf-8')
            print("  ✓ Creado README.md")
    
    def organize_files(self):
        """Organiza archivos en la nueva estructura (con confirmación)"""
        print("\n📂 ¿Organizar archivos en la nueva estructura?")
        
        moves = []
        for category, files in self.organization_plan.items():
            if category == 'main':
                continue  # No mover archivos principales
                
            for file in files:
                if file.parent == self.root:  # Solo archivos en raíz
                    if category == 'tests':
                        dest = self.root / 'tests' / file.name
                    else:
                        dest = self.root / 'src' / category / file.name
                    moves.append((file, dest))
        
        if moves:
            print(f"\nSe moverán {len(moves)} archivos:")
            for src, dest in moves[:5]:
                print(f"  {src.name} → {dest.parent.relative_to(self.root)}/")
            
            if len(moves) > 5:
                print(f"  ... y {len(moves) - 5} más")
            
            response = input("\n¿Proceder con la reorganización? (s/n): ")
            if response.lower() == 's':
                for src, dest in moves:
                    try:
                        shutil.move(str(src), str(dest))
                        print(f"  ✓ Movido: {src.name}")
                    except Exception as e:
                        print(f"  ✗ Error moviendo {src.name}: {e}")
    
    def generate_requirements(self):
        """Genera requirements.txt basado en imports"""
        print("\n📋 Generando requirements.txt...")
        
        # Mapeo de imports a paquetes pip
        package_map = {
            'pandas': 'pandas>=2.0.0',
            'numpy': 'numpy>=1.24.0',
            'matplotlib': 'matplotlib>=3.7.0',
            'seaborn': 'seaborn>=0.12.0',
            'sklearn': 'scikit-learn>=1.3.0',
            'scipy': 'scipy>=1.10.0',
            'requests': 'requests>=2.31.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'asyncio': '',  # Built-in
            'yfinance': 'yfinance>=0.2.28',
            'ta': 'ta>=0.10.2',
            'ccxt': 'ccxt>=4.0.0',
            'backtrader': 'backtrader>=1.9.78.123',
            'pytest': 'pytest>=7.4.0',
            'python-dotenv': 'python-dotenv>=1.0.0'
        }
        
        found_packages = set()
        
        for file in self.project_files:
            try:
                content = file.read_text(encoding='utf-8')
                for package, pip_name in package_map.items():
                    if f'import {package}' in content or f'from {package}' in content:
                        if pip_name:  # No agregar built-ins
                            found_packages.add(pip_name)
            except:
                continue
        
        # Agregar esenciales
        found_packages.update([
            'python-dotenv>=1.0.0',
            'pyyaml>=6.0',
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0'
        ])
        
        requirements_path = self.root / "requirements.txt"
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(found_packages)))
        
        print(f"  ✓ Creado requirements.txt con {len(found_packages)} paquetes")

def main():
    print("🚀 Smart Trading Bot Organizer")
    print("="*50)
    
    organizer = SmartProjectOrganizer()
    
    # Analizar proyecto
    organizer.analyze_project()
    
    # Mostrar plan
    organizer.show_organization_plan()
    
    # Limpiar
    organizer.clean_project()
    
    # Verificar imports
    organizer.check_imports()
    
    # Crear estructura
    response = input("\n¿Crear estructura de carpetas recomendada? (s/n): ")
    if response.lower() == 's':
        organizer.create_project_structure()
        organizer.create_readme()
        organizer.generate_requirements()
    
    # Organizar archivos
    organizer.organize_files()
    
    print("\n✅ Proceso completado!")
    print("\n💡 Próximos pasos:")
    print("1. Revisar archivos movidos")
    print("2. Actualizar imports en los archivos")
    print("3. Ejecutar tests para verificar que todo funciona")
    print("4. Commitear cambios: git add . && git commit -m 'Reorganización del proyecto'")

if __name__ == "__main__":
    main()