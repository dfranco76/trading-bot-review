# src/parameter_optimizer.py
"""
Sistema para optimizar automáticamente los parámetros del bot
"""
import numpy as np
import pandas as pd
import itertools
from datetime import datetime, timedelta
import json

# Importar scipy solo si está disponible
try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ scipy no instalado. Algunas funciones de optimización no estarán disponibles.")
    print("   Instala con: pip install scipy")
    SCIPY_AVAILABLE = False

class ParameterOptimizer:
    def __init__(self, backtest_engine=None):
        self.backtest_engine = backtest_engine
        self.optimization_history = []
        
    def optimize_agent_weights(self, symbols, start_date, end_date):
        """Optimiza los pesos de cada agente"""
        if not SCIPY_AVAILABLE:
            print("❌ scipy no disponible. Usando optimización simple.")
            return self.simple_weight_optimization(symbols, start_date, end_date)
        
        # Definir límites para cada peso (0.5 a 1.5)
        bounds = [
            (0.5, 1.5),  # Momentum
            (0.5, 1.5),  # Mean Reversion
            (0.5, 1.5),  # Pattern Recognition
            (0.5, 1.5),  # Volume
            (0.5, 1.5),  # Sentiment
        ]
        
        def objective(weights):
            """Función objetivo a maximizar (Sharpe ratio)"""
            # Configurar pesos
            agent_weights = {
                'momentum': weights[0],
                'mean_reversion': weights[1],
                'pattern': weights[2],
                'volume': weights[3],
                'sentiment': weights[4]
            }
            
            # Ejecutar backtest con estos pesos
            results = self.run_backtest_with_weights(
                symbols, 
                agent_weights, 
                start_date, 
                end_date
            )
            
            # Calcular métrica a optimizar (Sharpe ratio promedio)
            sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results.values()]
            avg_sharpe = np.mean(sharpe_ratios)
            
            # Penalizar si hay muy pocos trades
            total_trades = sum(r['metrics']['total_trades'] for r in results.values())
            if total_trades < 10:
                avg_sharpe *= 0.5
            
            return -avg_sharpe  # Negativo porque minimizamos
        
        # Ejecutar optimización
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42
        )
        
        optimal_weights = {
            'momentum': result.x[0],
            'mean_reversion': result.x[1],
            'pattern': result.x[2],
            'volume': result.x[3],
            'sentiment': result.x[4]
        }
        
        return optimal_weights, -result.fun  # Sharpe ratio óptimo
    
    def simple_weight_optimization(self, symbols, start_date, end_date):
        """Optimización simple sin scipy"""
        print("🔧 Ejecutando optimización simple de pesos...")
        
        best_weights = None
        best_sharpe = -float('inf')
        
        # Probar algunas combinaciones predefinidas
        weight_options = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for weights in itertools.product(weight_options, repeat=5):
            agent_weights = {
                'momentum': weights[0],
                'mean_reversion': weights[1],
                'pattern': weights[2],
                'volume': weights[3],
                'sentiment': weights[4]
            }
            
            # Simular resultado (en producción, ejecutarías el backtest real)
            # Por ahora, usar una función simple de evaluación
            sharpe = self.evaluate_weights_simple(agent_weights)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = agent_weights
                
        return best_weights, best_sharpe
    
    def evaluate_weights_simple(self, weights):
        """Evaluación simple de pesos (simulada)"""
        # En producción, esto ejecutaría un backtest real
        # Por ahora, simular un score basado en los pesos
        
        # Favorecer balance entre agentes
        balance_score = 1 - np.std(list(weights.values()))
        
        # Favorecer momentum y mean reversion ligeramente
        preference_score = (weights['momentum'] * 1.1 + weights['mean_reversion'] * 1.1) / 2
        
        # Score combinado
        return balance_score * 0.5 + preference_score * 0.5
    
    def optimize_consensus_thresholds(self, symbols, start_date, end_date):
        """Optimiza umbrales de consenso"""
        
        # Definir espacio de búsqueda
        thresholds_to_test = {
            'consenso_fuerte': np.arange(0.50, 0.70, 0.05),
            'consenso_moderado': np.arange(0.35, 0.50, 0.05),
            'min_agentes': [1, 2, 3],
            'confianza_single': np.arange(0.70, 0.90, 0.05)
        }
        
        best_config = None
        best_score = -float('inf')
        
        # Grid search
        for params in itertools.product(*thresholds_to_test.values()):
            config = {
                'consenso_fuerte': params[0],
                'consenso_moderado': params[1],
                'min_agentes': params[2],
                'confianza_single': params[3]
            }
            
            # Validar configuración
            if config['consenso_moderado'] >= config['consenso_fuerte']:
                continue
            
            # Simular backtest (en producción sería real)
            score = self.evaluate_consensus_config(config)
            
            if score > best_score:
                best_score = score
                best_config = config
                
            print(f"Config: {config} -> Score: {score:.3f}")
        
        return best_config, best_score
    
    def evaluate_consensus_config(self, config):
        """Evalúa configuración de consenso (simulada)"""
        # En producción, ejecutaría backtest real
        # Por ahora, usar heurísticas
        
        # Favorecer umbrales balanceados
        balance = 1 - abs(config['consenso_fuerte'] - 0.6)
        
        # Favorecer mínimo de agentes moderado
        agent_score = 1.0 if config['min_agentes'] == 2 else 0.8
        
        # Score combinado
        return balance * 0.6 + agent_score * 0.4
    
    def optimize_risk_parameters(self, symbols, start_date, end_date):
        """Optimiza parámetros de gestión de riesgo"""
        
        risk_params = {
            'max_position_size': np.arange(0.15, 0.30, 0.05),
            'stop_loss_pct': np.arange(0.015, 0.030, 0.005),
            'take_profit_pct': np.arange(0.020, 0.050, 0.010),
            'max_daily_trades': [10, 15, 20]
        }
        
        results = []
        
        for params in itertools.product(*risk_params.values()):
            config = {
                'max_position_size': params[0],
                'stop_loss_pct': params[1],
                'take_profit_pct': params[2],
                'max_daily_trades': params[3]
            }
            
            # Validar ratio riesgo/beneficio
            if config['take_profit_pct'] < config['stop_loss_pct'] * 1.2:
                continue
            
            # Simular evaluación
            metrics = self.evaluate_risk_config(config)
            
            results.append({
                'config': config,
                'metrics': metrics,
                'score': metrics['risk_adjusted_return']
            })
        
        # Ordenar por score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            return results[0]['config'], results[0]['metrics']
        else:
            # Configuración por defecto si no hay resultados
            return {
                'max_position_size': 0.20,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'max_daily_trades': 15
            }, {'risk_adjusted_return': 0.5}
    
    def evaluate_risk_config(self, config):
        """Evalúa configuración de riesgo (simulada)"""
        # Ratio riesgo/beneficio
        rr_ratio = config['take_profit_pct'] / config['stop_loss_pct']
        
        # Favorecer ratios entre 1.5 y 2.0
        if 1.5 <= rr_ratio <= 2.0:
            rr_score = 1.0
        else:
            rr_score = 0.7
        
        # Favorecer position size moderado
        if 0.15 <= config['max_position_size'] <= 0.25:
            pos_score = 1.0
        else:
            pos_score = 0.8
        
        # Score combinado
        risk_adjusted_return = rr_score * 0.6 + pos_score * 0.4
        
        return {
            'var_95': -0.02,  # Simulado
            'sortino_ratio': 1.5,  # Simulado
            'calmar_ratio': 2.0,  # Simulado
            'risk_adjusted_return': risk_adjusted_return
        }
    
    def calculate_optimization_score(self, results):
        """Calcula score compuesto para optimización"""
        if not results:
            return 0
            
        metrics = []
        
        for symbol, data in results.items():
            m = data.get('metrics', {})
            
            # Score compuesto
            score = (
                m.get('sharpe_ratio', 0) * 0.3 +
                m.get('profit_factor', 1) * 0.2 +
                m.get('win_rate', 0) * 0.2 +
                (1 - abs(m.get('max_drawdown', 0)) / 100) * 0.2 +
                min(m.get('total_trades', 0) / 50, 1) * 0.1  # Premiar actividad
            )
            
            metrics.append(score)
        
        return np.mean(metrics) if metrics else 0
    
    def calculate_risk_metrics(self, results):
        """Calcula métricas de riesgo avanzadas"""
        if not results:
            return {
                'var_95': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'risk_adjusted_return': 0
            }
            
        all_returns = []
        all_drawdowns = []
        
        for symbol, data in results.items():
            equity_curve = data.get('equity_curve', [])
            if len(equity_curve) > 1:
                returns = pd.Series(equity_curve).pct_change().dropna()
                all_returns.extend(returns.tolist())
            all_drawdowns.append(data.get('metrics', {}).get('max_drawdown', 0))
        
        if not all_returns:
            return {
                'var_95': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'risk_adjusted_return': 0
            }
        
        # Calcular métricas
        returns_series = pd.Series(all_returns)
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns_series, 5) if len(returns_series) > 0 else 0
        
        # Sortino Ratio (solo volatilidad negativa)
        negative_returns = returns_series[returns_series < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 1
        sortino_ratio = (returns_series.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        total_return = np.mean([r.get('metrics', {}).get('total_return', 0) for r in results.values()])
        avg_max_dd = np.mean(all_drawdowns) if all_drawdowns else 1
        calmar_ratio = total_return / abs(avg_max_dd) if avg_max_dd != 0 else 0
        
        return {
            'var_95': var_95,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'risk_adjusted_return': sortino_ratio * calmar_ratio
        }
    
    def save_optimal_config(self, config, filename='optimal_config.json'):
        """Guarda configuración óptima"""
        config['optimized_at'] = datetime.now().isoformat()
        config['version'] = '1.0'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuración óptima guardada en {filename}")
    
    def run_full_optimization(self, symbols, lookback_days=180):
        """Ejecuta optimización completa del sistema"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        print("🔧 OPTIMIZACIÓN COMPLETA DEL SISTEMA")
        print("="*50)
        
        # 1. Optimizar pesos de agentes
        print("\n1️⃣ Optimizando pesos de agentes...")
        agent_weights, sharpe = self.optimize_agent_weights(symbols, start_date, end_date)
        print(f"   Sharpe óptimo: {sharpe:.2f}")
        print(f"   Pesos: {agent_weights}")
        
        # 2. Optimizar umbrales de consenso
        print("\n2️⃣ Optimizando umbrales de consenso...")
        consensus_config, score = self.optimize_consensus_thresholds(symbols, start_date, end_date)
        print(f"   Score óptimo: {score:.3f}")
        print(f"   Config: {consensus_config}")
        
        # 3. Optimizar parámetros de riesgo
        print("\n3️⃣ Optimizando parámetros de riesgo...")
        risk_config, risk_metrics = self.optimize_risk_parameters(symbols, start_date, end_date)
        print(f"   Risk-adjusted return: {risk_metrics['risk_adjusted_return']:.3f}")
        print(f"   Config: {risk_config}")
        
        # Combinar todo
        optimal_config = {
            'agent_weights': agent_weights,
            'consensus': consensus_config,
            'risk': risk_config,
            'performance_metrics': {
                'expected_sharpe': sharpe,
                'optimization_score': score,
                'risk_metrics': risk_metrics
            }
        }
        
        # Guardar
        self.save_optimal_config(optimal_config)
        
        return optimal_config
    
    # Métodos stub para cuando no hay backtest engine real
    def run_backtest_with_weights(self, symbols, weights, start_date, end_date):
        """Stub para backtest con pesos específicos"""
        # En producción, esto ejecutaría un backtest real
        # Por ahora, retornar datos simulados
        results = {}
        for symbol in symbols:
            results[symbol] = {
                'metrics': {
                    'sharpe_ratio': np.random.uniform(0.5, 2.0),
                    'profit_factor': np.random.uniform(0.8, 2.5),
                    'win_rate': np.random.uniform(0.4, 0.6),
                    'max_drawdown': np.random.uniform(-20, -5),
                    'total_trades': np.random.randint(10, 50),
                    'total_return': np.random.uniform(-10, 30)
                },
                'equity_curve': [200 + i + np.random.randn() * 5 for i in range(100)]
            }
        return results
    
    def run_backtest_with_config(self, symbols, config, start_date, end_date):
        """Stub para backtest con configuración específica"""
        return self.run_backtest_with_weights(symbols, {}, start_date, end_date)
    
    def run_backtest_with_risk_params(self, symbols, config, start_date, end_date):
        """Stub para backtest con parámetros de riesgo específicos"""
        return self.run_backtest_with_weights(symbols, {}, start_date, end_date)


# Ejemplo de uso sin dependencias
if __name__ == "__main__":
    print("🔧 OPTIMIZADOR DE PARÁMETROS")
    print("="*50)
    
    # Crear optimizador sin backtest engine (modo simulación)
    optimizer = ParameterOptimizer()
    
    # Símbolos de prueba
    test_symbols = ['NVDA', 'TSLA', 'PLTR']
    
    # Ejecutar optimización
    print("\n⚙️ Ejecutando optimización (modo simulación)...")
    optimal_config = optimizer.run_full_optimization(test_symbols, lookback_days=30)
    
    print("\n✅ Optimización completada!")
    print("\n📋 Configuración óptima encontrada:")
    print(json.dumps(optimal_config, indent=2))
    
    print("\n💡 Nota: Esta es una simulación. Para resultados reales:")
    print("   1. Instala scipy: pip install scipy")
    print("   2. Conecta un motor de backtesting real")
    print("   3. Ejecuta con datos históricos reales")