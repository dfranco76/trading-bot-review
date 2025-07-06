"""
Tests unitarios para el Enhanced Risk Manager.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import correcto
from risk_management.risk_manager import EnhancedRiskManager


class TestEnhancedRiskManager(unittest.TestCase):
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
        
        # Crear instancia del Enhanced Risk Manager
        self.risk_manager = EnhancedRiskManager(capital_inicial=200)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        self.patcher.stop()
    
    # ===== TESTS DE INICIALIZACIÓN =====
    def test_initialization(self):
        """Test de inicialización correcta"""
        self.assertEqual(self.risk_manager.capital_inicial, 200)
        self.assertIsInstance(self.risk_manager, EnhancedRiskManager)
    
    # ===== TESTS DE CAPITAL =====
    def test_get_capital_actual_con_datos(self):
        """Test obtener capital cuando hay datos en BD"""
        # Configurar mock
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {'capital': 250.50}
        ]
        
        capital = self.risk_manager.get_capital_actual()
        
        self.assertEqual(capital, 250.50)
        self.mock_supabase.table.assert_called_with('bot_status')
    
    def test_get_capital_actual_sin_datos(self):
        """Test obtener capital cuando no hay datos devuelve el inicial"""
        # Configurar mock sin datos
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = []
        
        capital = self.risk_manager.get_capital_actual()
        
        self.assertEqual(capital, 200.0)  # Debe retornar capital inicial
    
    # ===== TESTS DE MÉTRICAS DEL DÍA =====
    def test_get_metricas_dia_sin_trades(self):
        """Test métricas cuando no hay trades en el día"""
        # Configurar mock para la consulta de trades
        mock_table = Mock()
        mock_select = Mock()
        mock_gte = Mock()
        mock_execute = Mock()
        
        mock_execute.return_value.data = []
        mock_gte.return_value.execute = mock_execute
        mock_select.return_value.gte = mock_gte
        mock_table.return_value.select = mock_select
        
        self.mock_supabase.table = mock_table
        
        metricas = self.risk_manager.get_metricas_dia()
        
        self.assertEqual(metricas['trades_totales'], 0)
        self.assertEqual(metricas['trades_ganadores'], 0)
        self.assertEqual(metricas['trades_perdedores'], 0)
        self.assertEqual(metricas['pnl_total'], 0)
        self.assertEqual(metricas['pnl_porcentaje'], 0)
        self.assertEqual(metricas['win_rate'], 0)
    
    def test_get_metricas_dia_con_trades_mixtos(self):
        """Test métricas con trades ganadores y perdedores"""
        # Configurar mock con trades
        trades_data = [
            {'id': 1, 'pnl': 10},      # Ganador
            {'id': 2, 'pnl': -5},      # Perdedor
            {'id': 3, 'pnl': 15},      # Ganador
            {'id': 4, 'pnl': None},    # Sin cerrar
        ]
        
        mock_table = Mock()
        mock_table.return_value.select.return_value.gte.return_value.execute.return_value.data = trades_data
        self.mock_supabase.table = mock_table
        
        metricas = self.risk_manager.get_metricas_dia()
        
        self.assertEqual(metricas['trades_totales'], 4)
        self.assertEqual(metricas['trades_ganadores'], 2)
        self.assertEqual(metricas['trades_perdedores'], 1)
        self.assertEqual(metricas['pnl_total'], 20)  # 10 - 5 + 15
        self.assertEqual(metricas['pnl_porcentaje'], 10.0)  # 20/200 * 100
        self.assertEqual(metricas['win_rate'], 0.5)  # 2/4
    
    # ===== TESTS DE VERIFICACIÓN DE HORARIO =====
    @patch('risk_management.risk_manager.datetime')
    def test_verificar_horario_trading_horario_valido(self, mock_datetime):
        """Test horario válido de trading"""
        # Configurar mock para martes 16:00
        mock_now = Mock()
        mock_now.hour = 16
        mock_now.minute = 0
        mock_now.weekday.return_value = 1  # Martes
        mock_datetime.now.return_value = mock_now
        
        ok, mensaje = self.risk_manager.verificar_horario_trading()
        
        self.assertTrue(ok)
        self.assertIn("óptimo", mensaje)
    
    @patch('risk_management.risk_manager.datetime')
    def test_verificar_horario_trading_fin_de_semana(self, mock_datetime):
        """Test horario en fin de semana"""
        # Configurar mock para sábado
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # Sábado
        mock_datetime.now.return_value = mock_now
        
        ok, mensaje = self.risk_manager.verificar_horario_trading()
        
        self.assertFalse(ok)
        self.assertIn("cerrado", mensaje.lower())


if __name__ == '__main__':
    unittest.main()



