# tests/test_risk_management/test_enhanced_risk_manager.py
"""
Tests para EnhancedRiskManager con sus métodos reales
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, time

# Mock dependencies
sys.modules['utils'] = MagicMock()
sys.modules['utils.config'] = MagicMock()
sys.modules['supabase'] = MagicMock()
sys.modules['yfinance'] = MagicMock()

# Configure mocks
sys.modules['utils.config'].SUPABASE_URL = 'http://mock-url'
sys.modules['utils.config'].SUPABASE_KEY = 'mock-key'

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))


class TestEnhancedRiskManager(unittest.TestCase):
    """Tests para los métodos reales de EnhancedRiskManager"""
    
    def setUp(self):
        """Setup para cada test"""
        # Mock supabase client
        self.mock_supabase = MagicMock()
        sys.modules['supabase'].create_client.return_value = self.mock_supabase
        
        # Mock yfinance
        self.mock_yf = sys.modules['yfinance']
        
        # Default mock responses
        self.mock_supabase.table().select().execute.return_value.data = []
        self.mock_supabase.table().insert().execute.return_value = None
        
        # Import after mocks
        from risk_management.risk_manager import EnhancedRiskManager
        self.risk_manager = EnhancedRiskManager()
        
    def test_initialization(self):
        """Test que el risk manager se inicializa correctamente"""
        self.assertIsNotNone(self.risk_manager)
        # Verificar que detectó régimen de mercado
        self.assertIn(self.risk_manager.market_regime, ['bull', 'bear', 'neutral'])
        
    def test_get_capital_actual(self):
        """Test obtención del capital actual"""
        # Mock respuesta de la base de datos
        self.mock_supabase.table().select().order().limit().execute.return_value.data = [
            {'balance': 50000.0}
        ]
        
        capital = self.risk_manager.get_capital_actual()
        
        self.assertEqual(capital, 50000.0)
        # Verificar que se llamó a la base de datos correctamente
        self.mock_supabase.table().select.assert_called_with('balance')
        
    def test_get_capital_actual_sin_datos(self):
        """Test cuando no hay datos de capital"""
        # Mock respuesta vacía
        self.mock_supabase.table().select().order().limit().execute.return_value.data = []
        
        capital = self.risk_manager.get_capital_actual()
        
        # Debería devolver capital inicial por defecto
        self.assertEqual(capital, 100000.0)
        
    def test_detectar_regimen_mercado_bull(self):
        """Test detección de mercado alcista"""
        # Mock datos de mercado alcista
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110]  # Tendencia alcista clara
        })
        self.mock_yf.Ticker.return_value = mock_ticker
        
        regimen = self.risk_manager.detectar_regimen_mercado('BTC-USD')
        
        self.assertEqual(regimen, 'bull')
        
    def test_detectar_regimen_mercado_bear(self):
        """Test detección de mercado bajista"""
        # Mock datos de mercado bajista
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({
            'Close': [110, 108, 106, 104, 102, 100]  # Tendencia bajista clara
        })
        self.mock_yf.Ticker.return_value = mock_ticker
        
        regimen = self.risk_manager.detectar_regimen_mercado('BTC-USD')
        
        self.assertEqual(regimen, 'bear')
        
    def test_calcular_correlacion_real(self):
        """Test cálculo de correlación real entre activos"""
        # Mock datos de dos activos
        mock_download = pd.DataFrame({
            ('Close', 'BTC-USD'): [100, 102, 101, 103, 105],
            ('Close', 'ETH-USD'): [2000, 2040, 2020, 2060, 2100]
        })
        mock_download.columns = pd.MultiIndex.from_tuples(mock_download.columns)
        self.mock_yf.download.return_value = mock_download
        
        correlacion = self.risk_manager.calcular_correlacion_real('BTC-USD', 'ETH-USD')
        
        # Debería devolver un valor entre -1 y 1
        self.assertGreaterEqual(correlacion, -1)
        self.assertLessEqual(correlacion, 1)
        
    def test_calcular_correlacion_aproximada(self):
        """Test cálculo de correlación aproximada"""
        correlacion = self.risk_manager.calcular_correlacion_aproximada('BTC-USD', 'ETH-USD')
        
        # Para crypto similares, debería ser alta
        self.assertGreater(correlacion, 0.5)
        
        # Para activos diferentes
        correlacion = self.risk_manager.calcular_correlacion_aproximada('BTC-USD', 'GOLD')
        self.assertLess(correlacion, 0.5)
        
    def test_calcular_var_historico(self):
        """Test cálculo de Value at Risk histórico"""
        # Mock datos históricos
        mock_ticker = MagicMock()
        returns = pd.Series(np.random.normal(-0.001, 0.02, 100))  # Retornos simulados
        prices = pd.Series(100 * (1 + returns).cumprod())
        mock_ticker.history.return_value = pd.DataFrame({'Close': prices})
        self.mock_yf.Ticker.return_value = mock_ticker
        
        var_95 = self.risk_manager.calcular_var_historico(
            symbol='BTC-USD',
            monto=10000,
            confianza=0.95
        )
        
        # VaR debería ser negativo (pérdida potencial)
        self.assertLess(var_95, 0)
        # No debería ser más del 10% en condiciones normales
        self.assertGreater(var_95, -1000)
        
    def test_verificar_horario_avanzado(self):
        """Test verificación de horario de trading"""
        # Mock diferentes horas
        with patch('risk_management.risk_manager.datetime') as mock_datetime:
            # Horario de trading normal (10 AM)
            mock_datetime.now.return_value.time.return_value = time(10, 0)
            mock_datetime.now.return_value.weekday.return_value = 1  # Martes
            
            puede_operar, mensaje = self.risk_manager.verificar_horario_avanzado()
            self.assertTrue(puede_operar)
            
            # Fin de semana
            mock_datetime.now.return_value.weekday.return_value = 6  # Domingo
            
            puede_operar, mensaje = self.risk_manager.verificar_horario_avanzado()
            self.assertFalse(puede_operar)
            self.assertIn("fin de semana", mensaje.lower())
            
    def test_ajustar_limites_dinamicamente(self):
        """Test ajuste dinámico de límites según volatilidad"""
        # Mock volatilidad alta
        mock_ticker = MagicMock()
        prices = pd.Series([100, 105, 98, 107, 95, 110])  # Alta volatilidad
        mock_ticker.history.return_value = pd.DataFrame({'Close': prices})
        self.mock_yf.Ticker.return_value = mock_ticker
        
        # Guardar límites originales
        limite_original = self.risk_manager.max_riesgo_operacion
        
        # Ajustar límites
        self.risk_manager.ajustar_limites_dinamicamente('BTC-USD')
        
        # Con alta volatilidad, debería reducir límites
        self.assertLess(self.risk_manager.max_riesgo_operacion, limite_original)
        
    def test_evaluar_trade_avanzado_aprobado(self):
        """Test evaluación de trade que debería ser aprobado"""
        # Mock capital disponible
        self.mock_supabase.table().select().order().limit().execute.return_value.data = [
            {'balance': 100000.0}
        ]
        
        # Mock volatilidad normal
        mock_ticker = MagicMock()
        prices = pd.Series(np.random.normal(100, 1, 30))
        mock_ticker.history.return_value = pd.DataFrame({'Close': prices})
        self.mock_yf.Ticker.return_value = mock_ticker
        
        # Mock horario válido
        with patch('risk_management.risk_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.time.return_value = time(10, 0)
            mock_datetime.now.return_value.weekday.return_value = 1
            
            # Trade conservador
            resultado = self.risk_manager.evaluar_trade_avanzado(
                symbol='BTC-USD',
                tipo='long',
                monto=1000,  # 1% del capital
                stop_loss=0.02  # 2% stop loss
            )
            
            self.assertTrue(resultado['aprobar'])
            self.assertIn('riesgo_total', resultado)
            self.assertIn('var_95', resultado)
            
    def test_evaluar_trade_avanzado_rechazado_por_riesgo(self):
        """Test evaluación de trade rechazado por alto riesgo"""
        # Mock capital disponible
        self.mock_supabase.table().select().order().limit().execute.return_value.data = [
            {'balance': 100000.0}
        ]
        
        # Trade muy riesgoso
        resultado = self.risk_manager.evaluar_trade_avanzado(
            symbol='BTC-USD',
            tipo='long',
            monto=30000,  # 30% del capital (muy alto)
            stop_loss=0.05  # 5% stop loss
        )
        
        self.assertFalse(resultado['aprobar'])
        self.assertIn('excede', resultado['mensaje'].lower())
        
    def test_generar_reporte_riesgo_avanzado(self):
        """Test generación de reporte de riesgo completo"""
        # Mock datos necesarios
        self.mock_supabase.table().select().order().limit().execute.return_value.data = [
            {'balance': 100000.0}
        ]
        
        # Mock posiciones abiertas
        self.mock_supabase.table().select().eq().execute.return_value.data = [
            {
                'symbol': 'BTC-USD',
                'entry_price': 50000,
                'current_price': 51000,
                'quantity': 0.1,
                'type': 'long'
            }
        ]
        
        # Generar reporte
        reporte = self.risk_manager.generar_reporte_riesgo_avanzado()
        
        # Verificar estructura del reporte
        self.assertIn('fecha', reporte)
        self.assertIn('capital_actual', reporte)
        self.assertIn('exposicion_total', reporte)
        self.assertIn('posiciones_abiertas', reporte)
        self.assertIn('regime_mercado', reporte)
        self.assertIn('metricas_riesgo', reporte)
        
        # Verificar que las métricas tienen valores
        self.assertIsNotNone(reporte['capital_actual'])
        self.assertEqual(reporte['regime_mercado'], self.risk_manager.market_regime)


class TestRiskManagerIntegrationScenarios(unittest.TestCase):
    """Tests de escenarios de integración"""
    
    def setUp(self):
        """Setup para tests de integración"""
        # Configurar mocks
        sys.modules['supabase'].create_client.return_value = MagicMock()
        from risk_management.risk_manager import EnhancedRiskManager
        self.risk_manager = EnhancedRiskManager()
        
    def test_escenario_volatilidad_extrema(self):
        """Test comportamiento en volatilidad extrema"""
        # Simular volatilidad extrema
        with patch.object(self.risk_manager, 'calcular_var_historico') as mock_var:
            mock_var.return_value = -5000  # VaR muy alto (5% del portfolio)
            
            # El sistema debería rechazar trades
            resultado = self.risk_manager.evaluar_trade_avanzado(
                'BTC-USD', 'long', 10000, 0.02
            )
            
            self.assertFalse(resultado['aprobar'])
            
    def test_escenario_cambio_regimen_mercado(self):
        """Test adaptación a cambio de régimen de mercado"""
        # Empezar en bull
        self.risk_manager.market_regime = 'bull'
        limite_bull = self.risk_manager.max_riesgo_operacion
        
        # Cambiar a bear
        with patch.object(self.risk_manager, 'detectar_regimen_mercado') as mock_regime:
            mock_regime.return_value = 'bear'
            self.risk_manager.market_regime = 'bear'
            self.risk_manager.ajustar_limites_dinamicamente('BTC-USD')
            
            # Límites deberían ser más conservadores en bear
            self.assertLess(self.risk_manager.max_riesgo_operacion, limite_bull)


if __name__ == '__main__':
    unittest.main(verbosity=2)