# circuit_breaker_system.py
"""
Sistema completo de circuit breakers y mecanismos de seguridad
Protege contra pérdidas catastróficas y fallos del sistema
"""
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import json
import logging
from dataclasses import dataclass, asdict
import os

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit broken, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 2
    half_open_max_calls: int = 3

@dataclass
class TradingLimits:
    max_daily_loss: float = 100.0
    max_position_loss: float = 50.0
    max_consecutive_losses: int = 3
    max_trades_per_hour: int = 20
    max_trades_per_day: int = 50
    max_error_rate: float = 0.1
    min_win_rate: float = 0.3
    max_drawdown: float = 0.15

class CircuitBreaker:
    """Circuit breaker individual para un servicio"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        
    def call(self, func: Callable, *args, **kwargs):
        """Ejecuta función protegida por circuit breaker"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(f"Circuit breaker {self.name} half-open call limit reached")
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Registra llamada exitosa"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self):
        """Registra fallo"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} OPEN after half-open failure")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} OPEN after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Verifica si debe intentar reset"""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.config.timeout_seconds))
    
    def get_state(self) -> Dict:
        """Retorna estado actual"""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
            }

class TradingCircuitBreakerSystem:
    """Sistema completo de circuit breakers para trading"""
    
    def __init__(self, limits: TradingLimits = None):
        self.limits = limits or TradingLimits()
        self.breakers = {}
        self.emergency_stop = False
        self.daily_stats = {
            'trades': 0,
            'errors': 0,
            'losses': 0,
            'consecutive_losses': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }
        self.position_stats = {}
        self.alerts = []
        self.lock = threading.Lock()
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TradingCircuitBreaker")
        
        # Inicializar circuit breakers
        self._initialize_breakers()
        
        # Iniciar monitoreo
        self._start_monitoring()
    
    def _initialize_breakers(self):
        """Inicializa circuit breakers para diferentes servicios"""
        # Breaker para API de datos
        self.breakers['data_api'] = CircuitBreaker(
            'data_api',
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30)
        )
        
        # Breaker para ejecución de órdenes
        self.breakers['order_execution'] = CircuitBreaker(
            'order_execution',
            CircuitBreakerConfig(failure_threshold=2, timeout_seconds=60)
        )
        
        # Breaker para análisis ML
        self.breakers['ml_analysis'] = CircuitBreaker(
            'ml_analysis',
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=120)
        )
        
        # Breaker para base de datos
        self.breakers['database'] = CircuitBreaker(
            'database',
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60)
        )
    
    def _start_monitoring(self):
        """Inicia thread de monitoreo continuo"""
        def monitor():
            while True:
                self._check_system_health()
                time.sleep(60)  # Check every minute
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def check_trade_allowed(self, trade_info: Dict) -> Tuple[bool, str]:
        """Verifica si un trade está permitido"""
        with self.lock:
            # Check emergency stop
            if self.emergency_stop:
                return False, "EMERGENCY STOP ACTIVE"
            
            # Check daily loss limit
            if abs(self.daily_stats['total_pnl']) > self.limits.max_daily_loss:
                self._trigger_emergency_stop("Daily loss limit exceeded")
                return False, f"Daily loss limit exceeded: ${abs(self.daily_stats['total_pnl']):.2f}"
            
            # Check consecutive losses
            if self.daily_stats['consecutive_losses'] >= self.limits.max_consecutive_losses:
                return False, f"Consecutive losses limit reached: {self.daily_stats['consecutive_losses']}"
            
            # Check trades per hour
            hour_trades = self._count_recent_trades(minutes=60)
            if hour_trades >= self.limits.max_trades_per_hour:
                return False, f"Hourly trade limit reached: {hour_trades}"
            
            # Check trades per day
            if self.daily_stats['trades'] >= self.limits.max_trades_per_day:
                return False, f"Daily trade limit reached: {self.daily_stats['trades']}"
            
            # Check win rate if enough trades
            if self.daily_stats['trades'] > 10:
                win_rate = self._calculate_win_rate()
                if win_rate < self.limits.min_win_rate:
                    return False, f"Win rate too low: {win_rate:.1%}"
            
            # Check position-specific limits
            symbol = trade_info.get('symbol')
            if symbol in self.position_stats:
                pos_loss = self.position_stats[symbol].get('unrealized_pnl', 0)
                if abs(pos_loss) > self.limits.max_position_loss:
                    return False, f"Position loss limit exceeded for {symbol}: ${abs(pos_loss):.2f}"
            
            # Check circuit breakers
            if any(breaker.state == CircuitState.OPEN for breaker in self.breakers.values()):
                open_breakers = [name for name, breaker in self.breakers.items() 
                               if breaker.state == CircuitState.OPEN]
                return False, f"Circuit breakers OPEN: {', '.join(open_breakers)}"
            
            return True, "Trade allowed"
    
    def record_trade_result(self, trade_id: str, result: Dict):
        """Registra resultado de un trade"""
        with self.lock:
            self.daily_stats['trades'] += 1
            
            pnl = result.get('pnl', 0)
            self.daily_stats['total_pnl'] += pnl
            
            if pnl < 0:
                self.daily_stats['losses'] += 1
                self.daily_stats['consecutive_losses'] += 1
            else:
                self.daily_stats['consecutive_losses'] = 0
            
            # Update position stats
            symbol = result.get('symbol')
            if symbol:
                if symbol not in self.position_stats:
                    self.position_stats[symbol] = {
                        'trades': 0,
                        'pnl': 0,
                        'unrealized_pnl': 0
                    }
                
                self.position_stats[symbol]['trades'] += 1
                self.position_stats[symbol]['pnl'] += pnl
            
            # Check for drawdown
            self._update_drawdown()
            
            # Log significant losses
            if pnl < -20:
                self._create_alert(
                    'significant_loss',
                    f"Significant loss on {symbol}: ${pnl:.2f}",
                    'high'
                )
    
    def record_error(self, error_type: str, error_msg: str):
        """Registra un error del sistema"""
        with self.lock:
            self.daily_stats['errors'] += 1
            
            error_rate = self.daily_stats['errors'] / max(self.daily_stats['trades'], 1)
            if error_rate > self.limits.max_error_rate:
                self._trigger_emergency_stop(f"Error rate too high: {error_rate:.1%}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Activa parada de emergencia"""
        self.emergency_stop = True
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Create critical alert
        self._create_alert(
            'emergency_stop',
            f"Emergency stop activated: {reason}",
            'critical'
        )
        
        # Notify all connected systems
        self._notify_emergency_stop(reason)
    
    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Crea una alerta"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"ALERT [{severity}] {alert_type}: {message}")
        
        # Send notification if critical
        if severity == 'critical':
            self._send_critical_notification(alert)
    
    def _send_critical_notification(self, alert: Dict):
        """Envía notificación crítica"""
        # Implement notification logic (email, SMS, etc.)
        pass
    
    def _notify_emergency_stop(self, reason: str):
        """Notifica a todos los sistemas de la parada de emergencia"""
        # Implement notification to all trading systems
        pass
    
    def _check_system_health(self):
        """Verifica salud general del sistema"""
        try:
            # Check drawdown
            if self.daily_stats['max_drawdown'] > self.limits.max_drawdown:
                self._trigger_emergency_stop(f"Max drawdown exceeded: {self.daily_stats['max_drawdown']:.1%}")
            
            # Check if trading should resume
            if self.emergency_stop:
                if self._should_resume_trading():
                    self.resume_trading()
            
            # Clean old alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts 
                          if datetime.fromisoformat(a['timestamp']) > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
    
    def _should_resume_trading(self) -> bool:
        """Determina si se debe resumir el trading"""
        # Check if enough time has passed
        if not self.alerts:
            return False
        
        last_critical = None
        for alert in reversed(self.alerts):
            if alert['severity'] == 'critical':
                last_critical = datetime.fromisoformat(alert['timestamp'])
                break
        
        if last_critical:
            time_since_critical = datetime.now() - last_critical
            if time_since_critical < timedelta(hours=1):
                return False
        
        # Check if conditions have improved
        if abs(self.daily_stats['total_pnl']) < self.limits.max_daily_loss * 0.8:
            return True
        
        return False
    
    def resume_trading(self):
        """Resume trading after emergency stop"""
        self.emergency_stop = False
        self.logger.info("Trading resumed after emergency stop")
        self._create_alert('trading_resumed', 'Trading has been resumed', 'info')
    
    def _count_recent_trades(self, minutes: int) -> int:
        """Cuenta trades recientes"""
        # Implement counting logic based on trade history
        return 0  # Placeholder
    
    def _calculate_win_rate(self) -> float:
        """Calcula win rate"""
        total = self.daily_stats['trades']
        if total == 0:
            return 0
        
        wins = total - self.daily_stats['losses']
        return wins / total
    
    def _update_drawdown(self):
        """Actualiza drawdown máximo"""
        # Implement drawdown calculation
        pass
    
    def get_system_status(self) -> Dict:
        """Retorna estado completo del sistema"""
        with self.lock:
            breaker_states = {name: breaker.get_state() 
                            for name, breaker in self.breakers.items()}
            
            return {
                'emergency_stop': self.emergency_stop,
                'daily_stats': self.daily_stats.copy(),
                'circuit_breakers': breaker_states,
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'trading_allowed': not self.emergency_stop,
                'health_score': self._calculate_health_score()
            }
    
    def _calculate_health_score(self) -> float:
        """Calcula score de salud del sistema (0-100)"""
        score = 100.0
        
        # Penalize for losses
        loss_ratio = abs(self.daily_stats['total_pnl']) / self.limits.max_daily_loss
        score -= loss_ratio * 30
        
        # Penalize for errors
        if self.daily_stats['trades'] > 0:
            error_rate = self.daily_stats['errors'] / self.daily_stats['trades']
            score -= error_rate * 20
        
        # Penalize for open circuit breakers
        open_breakers = sum(1 for b in self.breakers.values() 
                           if b.state == CircuitState.OPEN)
        score -= open_breakers * 10
        
        # Penalize for low win rate
        if self.daily_stats['trades'] > 5:
            win_rate = self._calculate_win_rate()
            if win_rate < 0.4:
                score -= (0.4 - win_rate) * 50
        
        return max(0, min(100, score))
    
    def reset_daily_stats(self):
        """Reset estadísticas diarias"""
        with self.lock:
            self.daily_stats = {
                'trades': 0,
                'errors': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
            self.position_stats.clear()
            self.logger.info("Daily stats reset")
    
    def save_state(self, filepath: str):
        """Guarda estado del sistema"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'emergency_stop': self.emergency_stop,
            'daily_stats': self.daily_stats,
            'position_stats': self.position_stats,
            'alerts': self.alerts,
            'limits': asdict(self.limits)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Carga estado previo del sistema"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.emergency_stop = state.get('emergency_stop', False)
            self.daily_stats = state.get('daily_stats', self.daily_stats)
            self.position_stats = state.get('position_stats', {})
            self.alerts = state.get('alerts', [])
            
            self.logger.info("State loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")


# Failsafe mechanisms
class FailsafeManager:
    """Gestor de mecanismos de seguridad adicionales"""
    
    def __init__(self):
        self.killswitch_active = False
        self.safe_mode = False
        self.recovery_procedures = {}
        
    def activate_killswitch(self, reason: str):
        """Activa killswitch - detiene TODO el trading inmediatamente"""
        self.killswitch_active = True
        logging.critical(f"KILLSWITCH ACTIVATED: {reason}")
        
        # Cancel all pending orders
        self._cancel_all_orders()
        
        # Close all positions if configured
        if os.getenv('CLOSE_ON_KILLSWITCH', 'false').lower() == 'true':
            self._close_all_positions()
    
    def enter_safe_mode(self):
        """Entra en modo seguro - trading limitado"""
        self.safe_mode = True
        logging.warning("System entering SAFE MODE")
        
        # In safe mode:
        # - Only closing positions allowed
        # - Reduced position sizes
        # - Conservative strategies only
        # - Increased monitoring
    
    def _cancel_all_orders(self):
        """Cancela todas las órdenes pendientes"""
        # Implement order cancellation
        pass
    
    def _close_all_positions(self):
        """Cierra todas las posiciones abiertas"""
        # Implement position closing
        pass
    
    def register_recovery_procedure(self, error_type: str, procedure: Callable):
        """Registra procedimiento de recuperación para tipo de error"""
        self.recovery_procedures[error_type] = procedure
    
    def attempt_recovery(self, error_type: str, error_context: Dict) -> bool:
        """Intenta recuperación automática"""
        if error_type in self.recovery_procedures:
            try:
                procedure = self.recovery_procedures[error_type]
                return procedure(error_context)
            except Exception as e:
                logging.error(f"Recovery procedure failed: {e}")
                return False
        return False


if __name__ == "__main__":
    # Example usage
    circuit_system = TradingCircuitBreakerSystem()
    
    # Test trade validation
    test_trade = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.0
    }
    
    allowed, reason = circuit_system.check_trade_allowed(test_trade)
    print(f"Trade allowed: {allowed} - {reason}")
    
    # Get system status
    status = circuit_system.get_system_status()
    print(f"System health score: {status['health_score']:.1f}/100")
    
    # Test circuit breaker
    data_breaker = circuit_system.breakers['data_api']
    
    def failing_api_call():
        raise Exception("API Error")
    
    # This will open the circuit breaker after failures
    for i in range(5):
        try:
            data_breaker.call(failing_api_call)
        except:
            print(f"Call {i+1} failed")
    
    print(f"Data API breaker state: {data_breaker.state.value}")