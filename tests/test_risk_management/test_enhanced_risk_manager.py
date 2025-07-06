"""
Tests unitarios para el Enhanced Risk Manager.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

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
        
        # Mock para yfinance también
        self.yf_patcher = patch('risk_management.risk_manager.yf')
        self.mock_yf = self.yf_patcher.start()
        
        # Configurar respuesta por defecto para yfinance
        mock_ticker = Mock()
        mock_history = Mock()
        mock_data = MagicMock()
        
        # Simular datos de precios
        mock_data.__len__.return_value = 100
        mock_data.__getitem__.return_value = mock_data
        mock_data.iloc = MagicMock()
        mock_data.iloc.__getitem__.return_value = 50.0
        mock_data.rolling.return_value.mean.return_value = mock_data
        mock_data.rolling.return_value.std.return_value = mock_data
        mock_data.ewm.return_value.mean.return_value = mock_data
        mock_data.pct_change.return_value = mock_data
        mock_data.diff.return_value = mock_data
        mock_data.where.return_value = mock_data
        mock_data.dropna.return_value = mock_data
        
        # Configurar el DataFrame con índices apropiados
        mock_data.__class__ = MagicMock
        mock_data.__class__.__name__ = 'DataFrame'
        
        # Simular columnas del DataFrame
        mock_close = MagicMock()
        mock_close.iloc = MagicMock()
        mock_close.iloc.__getitem__.return_value = 50.0
        mock_close.rolling.return_value.mean.return_value = mock_close
        mock_close.rolling.return_value.std.return_value = mock_close
        mock_close.ewm.return_value.mean.return_value = mock_close
        mock_close.pct_change.return_value = mock_close
        mock_close.diff.return_value = mock_close
        
        mock_high = MagicMock()
        mock_high.iloc = MagicMock()
        mock_high.iloc.__getitem__.return_value = 52.0
        
        mock_low = MagicMock()
        mock_low.iloc = MagicMock()
        mock_low.iloc.__getitem__.return_value = 48.0
        
        # Asignar las columnas
        mock_data.__getitem__.side_effect = lambda key: {
            'Close': mock_close,
            'High': mock_high,
            'Low': mock_low
        }.get(key, mock_data)
        
        mock_history.return_value = mock_data
        mock_ticker.history = mock_history
        self.mock_yf.Ticker.return_value = mock_ticker
        
        # Configurar respuesta por defecto para capital
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {'capital': 200}
        ]
        
        # Crear instancia del Enhanced Risk Manager
        self.risk_manager = EnhancedRiskManager(capital_inicial=200)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        self.patcher.stop()
        self.yf_patcher.stop()
    
    # ===== TESTS DE INICIALIZACIÓN =====
    def test_initialization(self):
        """Test de inicialización correcta"""
        self.assertEqual(self.risk_manager.capital_inicial, 200)
        self.assertIsInstance(self.risk_manager, EnhancedRiskManager)
        self.assertIsNotNone(self.risk_manager.base_limits)
        self.assertIsNotNone(self.risk_manager.adaptive_factors)
    
    # ===== TESTS DE CAPITAL =====
    def test_get_capital_actual_con_datos(self):
        """Test obtener capital cuando hay datos en BD"""
        # Configurar mock
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {'capital': 250.50}
        ]
        
        # Limpiar cache si existe
        if hasattr(self.risk_manager, '_capital_cache'):
            delattr(self.risk_manager, '_capital_cache')
        if hasattr(self.risk_manager, '_capital_cache_time'):
            delattr(self.risk_manager, '_capital_cache_time')
        
        capital = self.risk_manager.get_capital_actual()
        
        self.assertEqual(capital, 250.50)
        self.mock_supabase.table.assert_called_with('bot_status')
    
    def test_get_capital_actual_sin_datos(self):
        """Test obtener capital cuando no hay datos devuelve el inicial"""
        # Configurar mock sin datos
        self.mock_supabase.table.return_value.select.return_value.execute.return_value.data = []
        
        # Limpiar cache
        if hasattr(self.risk_manager, '_capital_cache'):
            delattr(self.risk_manager, '_capital_cache')
        if hasattr(self.risk_manager, '_capital_cache_time'):
            delattr(self.risk_manager, '_capital_cache_time')
        
        capital = self.risk_manager.get_capital_actual()
        
        self.assertEqual(capital, 200.0)  # Debe retornar capital inicial
    
    # ===== TESTS DE MÉTRICAS DEL DÍA =====
    def test_get_metricas_dia_sin_trades(self):
        """Test métricas cuando no hay trades en el día"""
        # Configurar mock para la consulta de trades
        self.mock_supabase.table.return_value.select.return_value.gte.return_value.execute.return_value.data = []
        
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
        
        self.mock_supabase.table.return_value.select.return_value.gte.return_value.execute.return_value.data = trades_data
        
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
        mock_now.hour = 16
        mock_now.minute = 0
        mock_now.weekday.return_value = 5  # Sábado
        mock_datetime.now.return_value = mock_now
        
        ok, mensaje = self.risk_manager.verificar_horario_trading()
        
        self.assertFalse(ok)
        self.assertIn("Fin de semana", mensaje)
    
    # ===== TESTS ADICIONALES =====
    def test_detectar_regimen_mercado(self):
        """Test detección de régimen de mercado"""
        # Este test verifica que el método existe y retorna la estructura esperada
        regime = self.risk_manager.market_regime
        
        self.assertIn('regime', regime)
        self.assertIn('confidence', regime)
        self.assertIn('volatility', regime)
        self.assertIn(regime['regime'], ['bull', 'bear', 'sideways', 'volatile', 'unknown'])
    
    def test_calcular_exposicion_actual_sin_posiciones(self):
        """Test calcular exposición sin posiciones abiertas"""
        # Configurar mock sin posiciones
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        
        exposicion = self.risk_manager.calcular_exposicion_actual()
        
        self.assertEqual(exposicion['exposicion_total'], 0)
        self.assertEqual(exposicion['exposicion_total_pct'], 0)
        self.assertEqual(len(exposicion['simbolos_activos']), 0)
    
    def test_calcular_exposicion_actual_con_posiciones(self):
        """Test calcular exposición con posiciones abiertas"""
        # Configurar mock con posiciones
        posiciones_data = [
            {'symbol': 'AAPL', 'quantity': 10, 'price': 150, 'status': 'OPEN'},
            {'symbol': 'MSFT', 'quantity': 5, 'price': 300, 'status': 'OPEN'}
        ]
        
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = posiciones_data
        
        exposicion = self.risk_manager.calcular_exposicion_actual()
        
        # Verificar cálculos
        # AAPL: 10 * 150 = 1500
        # MSFT: 5 * 300 = 1500
        # Total: 3000
        self.assertEqual(exposicion['exposicion_total'], 3000)
        self.assertEqual(exposicion['exposicion_total_pct'], 15.0)  # 3000/200 = 15
        self.assertEqual(len(exposicion['simbolos_activos']), 2)
        self.assertIn('AAPL', exposicion['simbolos_activos'])
        self.assertIn('MSFT', exposicion['simbolos_activos'])


if __name__ == '__main__':
    unittest.main()



