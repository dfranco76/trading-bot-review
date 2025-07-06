# tests/test_risk_management/test_risk_manager.py
"""
Tests unitarios para EnhancedRiskManager
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.risk_management.risk_manager import EnhancedRiskManager


class TestEnhancedRiskManager(unittest.TestCase):
    """Test suite para EnhancedRiskManager"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.config = {
            'max_position_size': 0.1,  # 10% máximo por posición
            'max_daily_loss': 0.05,     # 5% pérdida diaria máxima
            'stop_loss_percentage': 0.02, # 2% stop loss
            'max_open_positions': 3,
            'risk_per_trade': 0.01      # 1% riesgo por trade
        }
        self.risk_manager = EnhancedRiskManager(self.config)
        
    def test_initialization(self):
        """Test que el risk manager se inicializa correctamente"""
        self.assertEqual(self.risk_manager.config['max_position_size'], 0.1)
        self.assertEqual(self.risk_manager.config['max_daily_loss'], 0.05)
        self.assertEqual(self.risk_manager.daily_loss, 0)
        
    def test_calculate_position_size_basic(self):
        """Test cálculo básico de tamaño de posición"""
        account_balance = 10000
        risk_amount = 100  # 1% de riesgo
        stop_loss_distance = 0.02  # 2%
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance, risk_amount, stop_loss_distance
        )
        
        # Con $100 de riesgo y 2% de stop loss: 100 / 0.02 = $5000
        expected_size = 5000
        self.assertEqual(position_size, expected_size)
        
    def test_position_size_respects_max_limit(self):
        """Test que el tamaño de posición respeta el límite máximo"""
        account_balance = 10000
        risk_amount = 500  # 5% de riesgo (muy alto)
        stop_loss_distance = 0.01  # 1% stop loss
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance, risk_amount, stop_loss_distance
        )
        
        # Debería estar limitado al 10% del balance = $1000
        max_allowed = account_balance * self.config['max_position_size']
        self.assertLessEqual(position_size, max_allowed)
        
    def test_validate_trade_within_limits(self):
        """Test validación de trade dentro de límites"""
        trade = {
            'symbol': 'BTC/USDT',
            'size': 0.05,  # 5% del portfolio
            'side': 'buy',
            'price': 50000
        }
        
        is_valid, message = self.risk_manager.validate_trade(
            trade, 
            account_balance=10000,
            open_positions=1
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Trade validado correctamente")
        
    def test_validate_trade_exceeds_position_limit(self):
        """Test rechazo de trade que excede límite de posición"""
        trade = {
            'symbol': 'BTC/USDT',
            'size': 0.15,  # 15% del portfolio (excede el 10%)
            'side': 'buy',
            'price': 50000
        }
        
        is_valid, message = self.risk_manager.validate_trade(
            trade,
            account_balance=10000,
            open_positions=1
        )
        
        self.assertFalse(is_valid)
        self.assertIn("excede el tamaño máximo", message)
        
    def test_validate_trade_exceeds_open_positions(self):
        """Test rechazo cuando hay demasiadas posiciones abiertas"""
        trade = {
            'symbol': 'BTC/USDT',
            'size': 0.05,
            'side': 'buy',
            'price': 50000
        }
        
        is_valid, message = self.risk_manager.validate_trade(
            trade,
            account_balance=10000,
            open_positions=3  # Ya hay 3 posiciones (máximo)
        )
        
        self.assertFalse(is_valid)
        self.assertIn("posiciones abiertas", message)
        
    def test_daily_loss_limit(self):
        """Test que el límite de pérdida diaria funciona"""
        # Simular pérdida del 4%
        self.risk_manager.update_daily_loss(0.04)
        
        trade = {
            'symbol': 'BTC/USDT',
            'size': 0.02,  # Trade normal de 2%
            'side': 'buy',
            'price': 50000
        }
        
        # Con 4% de pérdida, un trade de 2% más podría exceder el límite de 5%
        is_valid, message = self.risk_manager.validate_trade(
            trade,
            account_balance=10000,
            open_positions=1,
            potential_loss=0.02  # 2% pérdida potencial
        )
        
        self.assertFalse(is_valid)
        self.assertIn("límite diario", message)
        
    def test_calculate_stop_loss_long_position(self):
        """Test cálculo de stop loss para posición larga"""
        entry_price = 100
        expected_stop = entry_price * (1 - self.config['stop_loss_percentage'])
        
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price, 
            position_side='long'
        )
        
        self.assertEqual(stop_loss, expected_stop)
        self.assertEqual(stop_loss, 98)  # 100 * 0.98
        
    def test_calculate_stop_loss_short_position(self):
        """Test cálculo de stop loss para posición corta"""
        entry_price = 100
        expected_stop = entry_price * (1 + self.config['stop_loss_percentage'])
        
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price,
            position_side='short'
        )
        
        self.assertEqual(stop_loss, expected_stop)
        self.assertEqual(stop_loss, 102)  # 100 * 1.02
        
    def test_risk_reward_ratio_validation(self):
        """Test validación de ratio riesgo/beneficio"""
        # Trade con buen ratio risk/reward (1:3)
        good_trade = {
            'entry_price': 100,
            'stop_loss': 98,    # 2% de riesgo
            'take_profit': 106  # 6% de beneficio
        }
        
        ratio = self.risk_manager.calculate_risk_reward_ratio(good_trade)
        self.assertEqual(ratio, 3.0)
        self.assertTrue(self.risk_manager.is_acceptable_risk_reward(ratio))
        
        # Trade con mal ratio risk/reward (1:0.5)
        bad_trade = {
            'entry_price': 100,
            'stop_loss': 98,   # 2% de riesgo
            'take_profit': 101 # 1% de beneficio
        }
        
        ratio = self.risk_manager.calculate_risk_reward_ratio(bad_trade)
        self.assertEqual(ratio, 0.5)
        self.assertFalse(self.risk_manager.is_acceptable_risk_reward(ratio))
        
    def test_portfolio_heat_check(self):
        """Test verificación de 'calor' del portfolio (exposición total)"""
        open_positions = [
            {'symbol': 'BTC/USDT', 'size': 0.03, 'unrealized_pnl': -0.01},
            {'symbol': 'ETH/USDT', 'size': 0.04, 'unrealized_pnl': 0.02},
            {'symbol': 'ADA/USDT', 'size': 0.02, 'unrealized_pnl': -0.005}
        ]
        
        total_exposure = self.risk_manager.calculate_portfolio_heat(open_positions)
        expected_exposure = 0.03 + 0.04 + 0.02  # 9% total
        
        self.assertEqual(total_exposure, expected_exposure)
        self.assertLess(total_exposure, 0.3)  # Debe ser menos del 30%
        
    def test_correlation_check(self):
        """Test verificación de correlación entre activos"""
        # Mock de correlaciones
        correlations = {
            ('BTC/USDT', 'ETH/USDT'): 0.85,  # Alta correlación
            ('BTC/USDT', 'ADA/USDT'): 0.6,   # Correlación media
            ('ETH/USDT', 'ADA/USDT'): 0.7    # Correlación media-alta
        }
        
        with patch.object(self.risk_manager, 'get_correlation', side_effect=lambda s1, s2: correlations.get((s1, s2), 0)):
            # No debería permitir nuevo trade de ETH si ya hay BTC (alta correlación)
            can_trade = self.risk_manager.check_correlation_limits(
                'ETH/USDT',
                existing_positions=['BTC/USDT']
            )
            self.assertFalse(can_trade)
            
            # Debería permitir trade de activo con baja correlación
            can_trade = self.risk_manager.check_correlation_limits(
                'GOLD/USD',
                existing_positions=['BTC/USDT']
            )
            self.assertTrue(can_trade)
            
    def test_reset_daily_counters(self):
        """Test reseteo de contadores diarios"""
        # Establecer algunos valores
        self.risk_manager.daily_loss = 0.03
        self.risk_manager.daily_trades = 5
        self.risk_manager.daily_wins = 3
        
        # Reset
        self.risk_manager.reset_daily_counters()
        
        # Verificar que se resetearon
        self.assertEqual(self.risk_manager.daily_loss, 0)
        self.assertEqual(self.risk_manager.daily_trades, 0)
        self.assertEqual(self.risk_manager.daily_wins, 0)
        
    @patch('src.risk_management.risk_manager.datetime')
    def test_emergency_stop_activation(self, mock_datetime):
        """Test activación de parada de emergencia"""
        # Simular pérdida severa
        self.risk_manager.update_daily_loss(0.08)  # 8% de pérdida
        
        # Verificar que se activa el emergency stop
        self.assertTrue(self.risk_manager.is_emergency_stop_active())
        
        # No debería permitir ningún trade nuevo
        trade = {
            'symbol': 'BTC/USDT',
            'size': 0.01,  # Trade pequeño
            'side': 'buy'
        }
        
        is_valid, message = self.risk_manager.validate_trade(
            trade,
            account_balance=10000,
            open_positions=0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Emergency stop", message)
        

class TestRiskManagerIntegration(unittest.TestCase):
    """Tests de integración para Risk Manager"""
    
    @patch('src.risk_management.risk_manager.market_data')
    def test_risk_manager_with_market_data(self, mock_market_data):
        """Test integración con datos de mercado reales"""
        # Mock de datos de mercado
        mock_market_data.get_price.return_value = 50000
        mock_market_data.get_volatility.return_value = 0.02
        
        config = {'max_position_size': 0.1}
        risk_manager = EnhancedRiskManager(config)
        
        # Test de ajuste dinámico por volatilidad
        adjusted_size = risk_manager.adjust_position_by_volatility(
            base_size=0.05,
            symbol='BTC/USDT'
        )
        
        # Con alta volatilidad, debería reducir el tamaño
        self.assertLess(adjusted_size, 0.05)


if __name__ == '__main__':
    unittest.main()