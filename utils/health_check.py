#!/usr/bin/env python3
"""
Health Check System for Trading Bot
Performs comprehensive system health verification following best practices.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'

# Add project to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class Config:
    """Configuration management using environment variables with defaults"""
    
    # Database configuration (with secure defaults)
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # API Keys (from environment or .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    YFINANCE_API_KEY = os.getenv('YFINANCE_API_KEY', '')
    
    # Trading configuration
    DEFAULT_CAPITAL = float(os.getenv('DEFAULT_CAPITAL', '10000'))
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))
    
    # System configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # Default to development for safety
    
    @classmethod
    def validate(cls) -> Tuple[bool, List[str]]:
        """Validate configuration and return status with error messages"""
        errors = []
        warnings = []
        
        # Check critical configurations
        if not cls.SUPABASE_URL and cls.ENVIRONMENT == 'production':
            errors.append("SUPABASE_URL not set for production environment")
        elif not cls.SUPABASE_URL:
            warnings.append("SUPABASE_URL not set (optional for development)")
            
        if not cls.ALPHA_VANTAGE_API_KEY:
            warnings.append("ALPHA_VANTAGE_API_KEY not set (some features may be limited)")
            
        return len(errors) == 0, errors, warnings


class HealthCheck:
    """Comprehensive system health check implementation"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'environment': Config.ENVIRONMENT,
            'checks': {},
            'overall_status': 'healthy'
        }
        
    def check_configuration(self) -> Dict[str, any]:
        """Check configuration validity"""
        logger.info("Checking configuration...")
        
        is_valid, errors, warnings = Config.validate()
        
        result = {
            'status': 'pass' if is_valid else 'fail',
            'errors': errors,
            'warnings': warnings,
            'environment': Config.ENVIRONMENT
        }
        
        if errors:
            logger.error(f"Configuration errors: {errors}")
        if warnings:
            logger.warning(f"Configuration warnings: {warnings}")
            
        return result
    
    def check_dependencies(self) -> Dict[str, any]:
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'yfinance',
            'pandas',
            'numpy',
            'ta',
            'alpaca_trade_api',
            'sklearn',
            'schedule'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        result = {
            'status': 'pass' if not missing_packages else 'fail',
            'installed': installed_packages,
            'missing': missing_packages,
            'total': len(required_packages)
        }
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
        
        return result
    
    def check_file_structure(self) -> Dict[str, any]:
        """Check if all required files and directories exist"""
        logger.info("Checking file structure...")
        
        required_paths = {
            'directories': [
                SRC_DIR / 'strategies',
                SRC_DIR / 'risk_management',
                SRC_DIR / 'utils',
                SRC_DIR / 'execution',
                SRC_DIR / 'monitoring',
                PROJECT_ROOT / 'logs',
                PROJECT_ROOT / 'data',
                PROJECT_ROOT / 'results'
            ],
            'files': [
                SRC_DIR / '__init__.py',
                SRC_DIR / 'main_bot.py',
                SRC_DIR / 'utils' / 'config.py',
                SRC_DIR / 'strategies' / 'sistema_multiagente.py',
                SRC_DIR / 'risk_management' / 'risk_manager.py'
            ]
        }
        
        missing_items = []
        existing_items = []
        
        for category, paths in required_paths.items():
            for path in paths:
                if path.exists():
                    existing_items.append(str(path.relative_to(PROJECT_ROOT)))
                else:
                    missing_items.append(str(path.relative_to(PROJECT_ROOT)))
        
        result = {
            'status': 'pass' if not missing_items else 'fail',
            'existing': len(existing_items),
            'missing': missing_items,
            'total': sum(len(paths) for paths in required_paths.values())
        }
        
        if missing_items:
            logger.warning(f"Missing paths: {missing_items[:5]}...")  # Show first 5
            
        return result
    
    def check_api_connectivity(self) -> Dict[str, any]:
        """Check connectivity to external APIs"""
        logger.info("Checking API connectivity...")
        
        results = {}
        
        # Check yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            results['yfinance'] = {
                'status': 'pass',
                'message': 'Connected successfully'
            }
        except Exception as e:
            results['yfinance'] = {
                'status': 'fail',
                'message': str(e)
            }
            logger.error(f"yfinance connection failed: {e}")
        
        # Check Alpha Vantage if API key is available
        if Config.ALPHA_VANTAGE_API_KEY:
            results['alpha_vantage'] = {
                'status': 'pass',
                'message': 'API key configured'
            }
        else:
            results['alpha_vantage'] = {
                'status': 'warning',
                'message': 'API key not configured'
            }
        
        overall_status = 'pass' if all(
            api['status'] == 'pass' for api in results.values()
        ) else 'warning'
        
        return {
            'status': overall_status,
            'apis': results
        }
    
    def check_database_connection(self) -> Dict[str, any]:
        """Check database connectivity if configured"""
        logger.info("Checking database connection...")
        
        if not Config.SUPABASE_URL:
            return {
                'status': 'skip',
                'message': 'Database not configured (optional for development)'
            }
        
        try:
            # Try to import and test connection
            from supabase import create_client, Client
            
            supabase: Client = create_client(
                Config.SUPABASE_URL,
                Config.SUPABASE_KEY
            )
            
            # Test with a simple query
            response = supabase.table('trades').select('count').limit(1).execute()
            
            return {
                'status': 'pass',
                'message': 'Database connected successfully'
            }
        except ImportError:
            return {
                'status': 'warning',
                'message': 'Supabase client not installed'
            }
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return {
                'status': 'fail',
                'message': str(e)
            }
    
    def check_permissions(self) -> Dict[str, any]:
        """Check file system permissions"""
        logger.info("Checking permissions...")
        
        directories_to_check = [
            PROJECT_ROOT / 'logs',
            PROJECT_ROOT / 'data',
            PROJECT_ROOT / 'results'
        ]
        
        permission_issues = []
        
        for directory in directories_to_check:
            if directory.exists():
                # Test write permissions
                test_file = directory / '.permission_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception as e:
                    permission_issues.append({
                        'path': str(directory.relative_to(PROJECT_ROOT)),
                        'error': str(e)
                    })
        
        return {
            'status': 'pass' if not permission_issues else 'fail',
            'issues': permission_issues
        }
    
    def run_all_checks(self) -> Dict[str, any]:
        """Run all health checks and compile results"""
        
        print("\n" + "="*60)
        print("🏥 SISTEMA DE HEALTH CHECK - TRADING BOT")
        print("="*60)
        
        checks = {
            'configuration': self.check_configuration(),
            'dependencies': self.check_dependencies(),
            'file_structure': self.check_file_structure(),
            'api_connectivity': self.check_api_connectivity(),
            'database': self.check_database_connection(),
            'permissions': self.check_permissions()
        }
        
        # Calculate overall status
        failed_checks = sum(1 for check in checks.values() 
                          if check['status'] == 'fail')
        warning_checks = sum(1 for check in checks.values() 
                           if check['status'] == 'warning')
        
        if failed_checks > 0:
            self.results['overall_status'] = 'unhealthy'
        elif warning_checks > 0:
            self.results['overall_status'] = 'degraded'
        else:
            self.results['overall_status'] = 'healthy'
        
        self.results['checks'] = checks
        self.results['summary'] = {
            'total_checks': len(checks),
            'passed': sum(1 for check in checks.values() 
                         if check['status'] == 'pass'),
            'failed': failed_checks,
            'warnings': warning_checks,
            'skipped': sum(1 for check in checks.values() 
                          if check['status'] == 'skip')
        }
        
        return self.results
    
    def print_results(self):
        """Print formatted health check results"""
        
        for check_name, result in self.results['checks'].items():
            status = result['status']
            icon = {
                'pass': '✅',
                'fail': '❌',
                'warning': '⚠️',
                'skip': '⏭️'
            }.get(status, '❓')
            
            print(f"\n{icon} {check_name.upper().replace('_', ' ')}:")
            
            if check_name == 'configuration':
                if result.get('errors'):
                    for error in result['errors']:
                        print(f"   ❌ {error}")
                if result.get('warnings'):
                    for warning in result['warnings']:
                        print(f"   ⚠️  {warning}")
                        
            elif check_name == 'dependencies':
                print(f"   📦 Installed: {len(result['installed'])}/{result['total']}")
                if result['missing']:
                    print(f"   ❌ Missing: {', '.join(result['missing'])}")
                    
            elif check_name == 'file_structure':
                print(f"   📁 Found: {result['existing']}/{result['total']}")
                if result['missing']:
                    print(f"   ❌ Missing: {len(result['missing'])} items")
                    
            elif check_name == 'api_connectivity':
                for api, status in result['apis'].items():
                    icon = '✅' if status['status'] == 'pass' else '❌'
                    print(f"   {icon} {api}: {status['message']}")
                    
            elif check_name == 'permissions':
                if result['issues']:
                    for issue in result['issues']:
                        print(f"   ❌ {issue['path']}: {issue['error']}")
                else:
                    print(f"   ✅ All directories writable")
            
            else:
                print(f"   💬 {result.get('message', 'Check completed')}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 RESUMEN:")
        summary = self.results['summary']
        print(f"   • Total de verificaciones: {summary['total_checks']}")
        print(f"   • ✅ Pasadas: {summary['passed']}")
        print(f"   • ❌ Fallidas: {summary['failed']}")
        print(f"   • ⚠️  Advertencias: {summary['warnings']}")
        print(f"   • ⏭️  Omitidas: {summary['skipped']}")
        
        # Overall status
        status_icon = {
            'healthy': '💚',
            'degraded': '💛',
            'unhealthy': '❤️'
        }.get(self.results['overall_status'], '❓')
        
        print(f"\n{status_icon} ESTADO GENERAL: {self.results['overall_status'].upper()}")
        print("="*60)
    
    def save_report(self, filename: Optional[str] = None):
        """Save health check report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_check_report_{timestamp}.json"
        
        report_path = PROJECT_ROOT / 'logs' / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        print(f"\n💾 Reporte guardado en: {report_path}")


def main():
    """Main entry point for health check"""
    try:
        health_check = HealthCheck()
        health_check.run_all_checks()
        health_check.print_results()
        
        # Save report if in production
        if Config.ENVIRONMENT == 'production':
            health_check.save_report()
        
        # Return appropriate exit code
        if health_check.results['overall_status'] == 'unhealthy':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error in health check: {e}", exc_info=True)
        print(f"\n❌ Error fatal en health check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()