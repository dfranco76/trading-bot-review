import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import correcto con el nombre REAL de la clase
from risk_management.risk_manager import EnhancedRiskManager


class TestRiskManager(unittest.TestCase):
    """Tests para el Enhanced Risk Manager"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Mock de Supabase
        self.mock_supabase = Mock()
        
        # Patch del cliente de Supabase
        self.patcher = patch('risk_management.risk_manager.supabase', self.mock_supabase)
        self.patcher.start()
        
        # Configurar respuesta por defecto para capital
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {'capital': 200}
        ]
        
        # Crear instancia del Risk Manager
        self.risk_manager = EnhancedRiskManager(capital_inicial=200)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        self.patcher.stop()
    
    def test_initialization_values(self):
        """Test que los valores se inicializan correctamente"""
        self.assertEqual(self.risk_manager.capital_inicial, 200)
        # Verificar que la instancia se creó correctamente
        self.assertIsNotNone(self.risk_manager)
        self.assertIsInstance(self.risk_manager, EnhancedRiskManager)


if __name__ == '__main__':
    unittest.main()
