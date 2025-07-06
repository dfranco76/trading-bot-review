# advanced_analytics_system.py
"""
Sistema avanzado de analytics con performance attribution,
factor analysis, y métricas sofisticadas
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PerformanceAttributionEngine:
    """Motor de atribución de performance"""
    
    def __init__(self):
        self.factor_models = {}
        self.risk_factors = ['market', 'size', 'value', 'momentum', 'quality']
        
    def brinson_attribution(self, portfolio_weights: pd.DataFrame,
                          benchmark_weights: pd.DataFrame,
                          portfolio_returns: pd.DataFrame,
                          benchmark_returns: pd.DataFrame) -> Dict:
        """Atribución de Brinson para selection y allocation"""
        
        # Alinear datos
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        portfolio_weights = portfolio_weights.loc[common_dates]
        benchmark_weights = benchmark_weights.loc[common_dates]
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calcular efectos
        allocation_effects = []
        selection_effects = []
        interaction_effects = []
        
        for date in common_dates:
            # Pesos
            wp = portfolio_weights.loc[date]
            wb = benchmark_weights.loc[date]
            
            # Retornos
            rp = portfolio_returns.loc[date]
            rb = benchmark_returns.loc[date]
            
            # Allocation effect: (wp - wb) * (rb - rb_total)
            rb_total = (wb * rb).sum()
            allocation = (wp - wb) * (rb - rb_total)
            
            # Selection effect: wb * (rp - rb)
            selection = wb * (rp - rb)
            
            # Interaction effect: (wp - wb) * (rp - rb)
            interaction = (wp - wb) * (rp - rb)
            
            allocation_effects.append(allocation)
            selection_effects.append(selection)
            interaction_effects.append(interaction)
        
        # Agregar resultados
        allocation_df = pd.DataFrame(allocation_effects, index=common_dates)
        selection_df = pd.DataFrame(selection_effects, index=common_dates)
        interaction_df = pd.DataFrame(interaction_effects, index=common_dates)
        
        # Total attribution
        total_allocation = allocation_df.sum()
        total_selection = selection_df.sum()
        total_interaction = interaction_df.sum()
        
        # Performance relativa
        portfolio_perf = (1 + portfolio_returns).prod() - 1
        benchmark_perf = (1 + benchmark_returns).prod() - 1
        active_return = portfolio_perf - benchmark_perf
        
        return {
            'total_active_return': active_return,
            'allocation_effect': total_allocation,
            'selection_effect': total_selection,
            'interaction_effect': total_interaction,
            'allocation_by_asset': allocation_df.sum().to_dict(),
            'selection_by_asset': selection_df.sum().to_dict(),
            'time_series': {
                'allocation': allocation_df.sum(axis=1),
                'selection': selection_df.sum(axis=1),
                'interaction': interaction_df.sum(axis=1)
            }
        }
    
    def factor_attribution(self, portfolio_returns: pd.Series,
                         factor_returns: pd.DataFrame) -> Dict:
        """Atribución basada en factores (Fama-French style)"""
        
        # Alinear datos
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        y = portfolio_returns.loc[common_dates]
        X = factor_returns.loc[common_dates]
        
        # Añadir constante
        X = sm.add_constant(X)
        
        # Regresión
        model = sm.OLS(y, X).fit()
        
        # Descomposición del retorno
        factor_contributions = {}
        for factor in factor_returns.columns:
            if factor in model.params:
                factor_contributions[factor] = (
                    model.params[factor] * X[factor].mean()
                )
        
        # Alpha
        alpha = model.params['const']
        
        # R-squared
        r_squared = model.rsquared
        
        # Residuales
        residuals = model.resid
        specific_risk = residuals.std()
        
        return {
            'alpha': alpha,
            'factor_loadings': model.params.drop('const').to_dict(),
            'factor_contributions': factor_contributions,
            'r_squared': r_squared,
            'specific_risk': specific_risk,
            'factor_pvalues': model.pvalues.to_dict(),
            'model_summary': model.summary()
        }
    
    def performance_decomposition(self, returns: pd.Series) -> Dict:
        """Descomposición completa de performance"""
        
        # Estadísticas básicas
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Descomposición por períodos
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        
        # Análisis de consistencia
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        
        # Rolling metrics
        rolling_returns = returns.rolling(window=252)
        rolling_vol = rolling_returns.std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns.mean() * 252) / (rolling_returns.std() * np.sqrt(252))
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': annualized_return / volatility,
            'positive_months': positive_months,
            'negative_months': negative_months,
            'best_month': best_month,
            'worst_month': worst_month,
            'monthly_win_rate': positive_months / (positive_months + negative_months),
            'rolling_metrics': {
                'volatility': rolling_vol,
                'sharpe': rolling_sharpe
            }
        }
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """Calcula Information Ratio"""
        active_returns = portfolio_returns - benchmark_returns
        
        if active_returns.std() == 0:
            return 0
        
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))
    
    def calculate_treynor_ratio(self, returns: pd.Series, 
                              market_returns: pd.Series,
                              risk_free_rate: float = 0.02) -> float:
        """Calcula Treynor Ratio"""
        # Beta
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance
        
        # Excess return
        excess_return = returns.mean() * 252 - risk_free_rate
        
        return excess_return / beta if beta != 0 else 0

class AdvancedRiskAnalytics:
    """Analytics avanzado de riesgo"""
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
        
    def calculate_expected_shortfall(self, returns: pd.Series, 
                                   confidence: float = 0.95) -> float:
        """Calcula Expected Shortfall (CVaR)"""
        var = returns.quantile(1 - confidence)
        conditional_returns = returns[returns <= var]
        
        return conditional_returns.mean() if len(conditional_returns) > 0 else var
    
    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict:
        """Métricas de riesgo de cola"""
        # Skewness y Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Tail ratios
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        
        gains_95 = returns[returns > percentile_95]
        losses_5 = returns[returns < percentile_5]
        
        gain_loss_ratio = abs(gains_95.mean() / losses_5.mean()) if len(losses_5) > 0 else np.inf
        
        # Best and worst returns
        best_day = returns.max()
        worst_day = returns.min()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': kurtosis - 3,
            'is_normal': jb_pvalue > 0.05,
            'jarque_bera_pvalue': jb_pvalue,
            'gain_loss_ratio_95_5': gain_loss_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'var_95': np.percentile(returns, 5),
            'cvar_95': self.calculate_expected_shortfall(returns, 0.95)
        }
    
    def stress_test_scenarios(self, portfolio_returns: pd.Series,
                            market_returns: pd.Series) -> Dict:
        """Pruebas de estrés con escenarios históricos"""
        
        # Identificar períodos de crisis
        crisis_periods = {
            'COVID-19': ('2020-02-20', '2020-03-23'),
            '2008 Crisis': ('2008-09-15', '2008-11-20'),
            'Dot-com': ('2000-03-10', '2001-10-09')
        }
        
        results = {}
        
        # Beta del portfolio
        beta = portfolio_returns.cov(market_returns) / market_returns.var()
        
        for crisis_name, (start, end) in crisis_periods.items():
            try:
                # Performance durante crisis
                crisis_market = market_returns.loc[start:end]
                crisis_portfolio = portfolio_returns.loc[start:end]
                
                if len(crisis_market) > 0:
                    market_drawdown = (crisis_market + 1).cumprod().iloc[-1] - 1
                    portfolio_drawdown = (crisis_portfolio + 1).cumprod().iloc[-1] - 1
                    
                    # Drawdown esperado basado en beta
                    expected_drawdown = beta * market_drawdown
                    
                    results[crisis_name] = {
                        'market_drawdown': market_drawdown,
                        'portfolio_drawdown': portfolio_drawdown,
                        'expected_drawdown': expected_drawdown,
                        'excess_drawdown': portfolio_drawdown - expected_drawdown,
                        'beta_during_crisis': crisis_portfolio.cov(crisis_market) / crisis_market.var()
                    }
            except:
                pass
        
        # Escenarios sintéticos
        synthetic_scenarios = {
            'Market Crash -20%': -0.20,
            'Market Crash -30%': -0.30,
            'Market Rally +20%': 0.20
        }
        
        for scenario_name, market_return in synthetic_scenarios.items():
            expected_portfolio_return = beta * market_return
            
            results[scenario_name] = {
                'market_return': market_return,
                'expected_portfolio_return': expected_portfolio_return
            }
        
        return results
    
    def calculate_rolling_correlations(self, returns_matrix: pd.DataFrame,
                                     window: int = 60) -> pd.DataFrame:
        """Calcula correlaciones rolling"""
        rolling_corr = returns_matrix.rolling(window=window).corr()
        
        # Extraer correlaciones promedio
        n_assets = len(returns_matrix.columns)
        avg_correlations = []
        
        for date in returns_matrix.index[window:]:
            corr_matrix = rolling_corr.loc[date]
            # Promedio de correlaciones (excluyendo diagonal)
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            avg_corr = corr_matrix.values[mask].mean()
            avg_correlations.append(avg_corr)
        
        return pd.Series(
            avg_correlations, 
            index=returns_matrix.index[window:],
            name='average_correlation'
        )

class MLPerformanceAnalytics:
    """Analytics específico para modelos ML"""
    
    def __init__(self):
        self.feature_importance_history = []
        self.prediction_accuracy_history = []
        
    def analyze_feature_importance_stability(self, 
                                           importance_history: List[Dict]) -> Dict:
        """Analiza estabilidad de importancia de features"""
        if not importance_history:
            return {}
        
        # Convertir a DataFrame
        importance_df = pd.DataFrame(importance_history)
        
        # Calcular estadísticas
        stability_metrics = {}
        
        for feature in importance_df.columns:
            values = importance_df[feature].dropna()
            if len(values) > 0:
                stability_metrics[feature] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'cv': values.std() / values.mean() if values.mean() > 0 else np.inf,
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                }
        
        # Ranking de estabilidad
        stability_ranking = sorted(
            stability_metrics.items(),
            key=lambda x: x[1]['cv']
        )
        
        return {
            'feature_stability': stability_metrics,
            'most_stable_features': [f[0] for f in stability_ranking[:10]],
            'least_stable_features': [f[0] for f in stability_ranking[-10:]]
        }
    
    def analyze_prediction_drift(self, predictions: pd.DataFrame,
                               actuals: pd.DataFrame) -> Dict:
        """Analiza drift en predicciones del modelo"""
        
        # Calcular error absoluto medio por ventana temporal
        errors = (predictions - actuals).abs()
        
        # Rolling MAE
        window = 50
        rolling_mae = errors.rolling(window=window).mean()
        
        # Detectar incrementos significativos en error
        mae_mean = rolling_mae.mean()
        mae_std = rolling_mae.std()
        drift_threshold = mae_mean + 2 * mae_std
        
        drift_periods = rolling_mae[rolling_mae > drift_threshold]
        
        # Analizar patrones de drift
        drift_analysis = {
            'has_drift': len(drift_periods) > 0,
            'drift_periods': drift_periods.index.tolist(),
            'average_mae': mae_mean,
            'max_mae': rolling_mae.max(),
            'drift_severity': (rolling_mae.max() - mae_mean) / mae_mean if mae_mean > 0 else 0
        }
        
        # Analizar por tipo de predicción
        if 'action' in predictions.columns:
            action_accuracy = {}
            for action in ['BUY', 'SELL', 'HOLD']:
                mask = predictions['action'] == action
                if mask.any():
                    action_accuracy[action] = (
                        predictions[mask] == actuals[mask]
                    ).mean()
            
            drift_analysis['action_accuracy'] = action_accuracy
        
        return drift_analysis
    
    def calculate_ml_sharpe_ratio(self, ml_returns: pd.Series,
                                baseline_returns: pd.Series) -> Dict:
        """Calcula Sharpe ratio específico para estrategias ML"""
        
        # Sharpe tradicional
        ml_sharpe = (ml_returns.mean() * 252) / (ml_returns.std() * np.sqrt(252))
        baseline_sharpe = (baseline_returns.mean() * 252) / (baseline_returns.std() * np.sqrt(252))
        
        # Information ratio (ML vs baseline)
        excess_returns = ml_returns - baseline_returns
        info_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        
        # Calmar ratio (return / max drawdown)
        ml_cumulative = (1 + ml_returns).cumprod()
        ml_drawdown = (ml_cumulative / ml_cumulative.cummax() - 1).min()
        calmar_ratio = (ml_returns.mean() * 252) / abs(ml_drawdown) if ml_drawdown != 0 else 0
        
        return {
            'ml_sharpe': ml_sharpe,
            'baseline_sharpe': baseline_sharpe,
            'sharpe_improvement': ml_sharpe - baseline_sharpe,
            'information_ratio': info_ratio,
            'calmar_ratio': calmar_ratio
        }

class PortfolioAnalytics:
    """Analytics avanzado de portfolio"""
    
    def __init__(self):
        self.attribution_engine = PerformanceAttributionEngine()
        self.risk_analytics = AdvancedRiskAnalytics()
        self.ml_analytics = MLPerformanceAnalytics()
        
    def generate_comprehensive_report(self, portfolio_data: Dict,
                                    benchmark_data: Dict,
                                    start_date: str,
                                    end_date: str) -> Dict:
        """Genera reporte comprehensivo de analytics"""
        
        # Convertir datos a Series/DataFrames
        portfolio_returns = pd.Series(portfolio_data['returns'])
        benchmark_returns = pd.Series(benchmark_data['returns'])
        
        report = {
            'period': {
                'start': start_date,
                'end': end_date,
                'trading_days': len(portfolio_returns)
            }
        }
        
        # 1. Performance básico
        report['performance'] = {
            'portfolio': self._calculate_basic_metrics(portfolio_returns),
            'benchmark': self._calculate_basic_metrics(benchmark_returns),
            'relative': {
                'excess_return': (
                    report['performance']['portfolio']['total_return'] -
                    report['performance']['benchmark']['total_return']
                ),
                'information_ratio': self.attribution_engine.calculate_information_ratio(
                    portfolio_returns, benchmark_returns
                )
            }
        }
        
        # 2. Risk metrics
        report['risk'] = {
            'portfolio_risk': self.risk_analytics.calculate_tail_risk_metrics(portfolio_returns),
            'benchmark_risk': self.risk_analytics.calculate_tail_risk_metrics(benchmark_returns),
            'relative_risk': {
                'beta': portfolio_returns.cov(benchmark_returns) / benchmark_returns.var(),
                'correlation': portfolio_returns.corr(benchmark_returns),
                'tracking_error': (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            }
        }
        
        # 3. Attribution (si hay datos de pesos)
        if 'weights' in portfolio_data and 'weights' in benchmark_data:
            report['attribution'] = self.attribution_engine.brinson_attribution(
                portfolio_data['weights'],
                benchmark_data['weights'],
                portfolio_returns,
                benchmark_returns
            )
        
        # 4. Stress testing
        report['stress_test'] = self.risk_analytics.stress_test_scenarios(
            portfolio_returns,
            benchmark_returns
        )
        
        # 5. Rolling analytics
        report['rolling_analytics'] = self._calculate_rolling_metrics(
            portfolio_returns,
            benchmark_returns
        )
        
        return report
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        """Calcula métricas básicas de performance"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1)
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_rolling_metrics(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 window: int = 60) -> Dict:
        """Calcula métricas rolling"""
        
        # Rolling correlation
        rolling_corr = portfolio_returns.rolling(window).corr(benchmark_returns)
        
        # Rolling beta
        rolling_cov = portfolio_returns.rolling(window).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        
        # Rolling Information Ratio
        active_returns = portfolio_returns - benchmark_returns
        rolling_ir = (
            active_returns.rolling(window).mean() * 252 /
            (active_returns.rolling(window).std() * np.sqrt(252))
        )
        
        return {
            'correlation': {
                'mean': rolling_corr.mean(),
                'std': rolling_corr.std(),
                'min': rolling_corr.min(),
                'max': rolling_corr.max(),
                'current': rolling_corr.iloc[-1] if len(rolling_corr) > 0 else None
            },
            'beta': {
                'mean': rolling_beta.mean(),
                'std': rolling_beta.std(),
                'min': rolling_beta.min(),
                'max': rolling_beta.max(),
                'current': rolling_beta.iloc[-1] if len(rolling_beta) > 0 else None
            },
            'information_ratio': {
                'mean': rolling_ir.mean(),
                'std': rolling_ir.std(),
                'min': rolling_ir.min(),
                'max': rolling_ir.max(),
                'current': rolling_ir.iloc[-1] if len(rolling_ir) > 0 else None
            }
        }
    
    def visualize_analytics(self, report: Dict, save_path: str = None):
        """Genera visualizaciones del reporte"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Portfolio Analytics Dashboard', fontsize=16)
        
        # 1. Cumulative returns
        ax = axes[0, 0]
        # Implementar gráfico de retornos acumulados
        ax.set_title('Cumulative Returns')
        
        # 2. Rolling Sharpe
        ax = axes[0, 1]
        # Implementar gráfico de Sharpe rolling
        ax.set_title('Rolling Sharpe Ratio')
        
        # 3. Drawdown
        ax = axes[0, 2]
        # Implementar gráfico de drawdown
        ax.set_title('Drawdown Analysis')
        
        # 4. Return distribution
        ax = axes[1, 0]
        # Implementar histograma de retornos
        ax.set_title('Return Distribution')
        
        # 5. Risk decomposition
        ax = axes[1, 1]
        # Implementar gráfico de descomposición de riesgo
        ax.set_title('Risk Decomposition')
        
        # 6. Attribution
        ax = axes[1, 2]
        # Implementar gráfico de atribución
        ax.set_title('Performance Attribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class RealTimeAnalytics:
    """Sistema de analytics en tiempo real"""
    
    def __init__(self):
        self.metrics_buffer = {}
        self.alert_thresholds = {
            'drawdown': -0.05,
            'volatility_spike': 2.0,
            'correlation_break': 0.3,
            'volume_anomaly': 3.0
        }
        
    def update_metrics(self, symbol: str, price: float, volume: int):
        """Actualiza métricas en tiempo real"""
        
        if symbol not in self.metrics_buffer:
            self.metrics_buffer[symbol] = {
                'prices': [],
                'volumes': [],
                'timestamps': [],
                'alerts': []
            }
        
        buffer = self.metrics_buffer[symbol]
        buffer['prices'].append(price)
        buffer['volumes'].append(volume)
        buffer['timestamps'].append(datetime.now())
        
        # Mantener solo últimos 1000 puntos
        if len(buffer['prices']) > 1000:
            buffer['prices'] = buffer['prices'][-1000:]
            buffer['volumes'] = buffer['volumes'][-1000:]
            buffer['timestamps'] = buffer['timestamps'][-1000:]
        
        # Calcular métricas si hay suficientes datos
        if len(buffer['prices']) >= 20:
            self._check_alerts(symbol)
    
    def _check_alerts(self, symbol: str):
        """Verifica condiciones de alerta"""
        buffer = self.metrics_buffer[symbol]
        prices = np.array(buffer['prices'])
        volumes = np.array(buffer['volumes'])
        
        # 1. Drawdown check
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        current_dd = drawdown[-1]
        
        if current_dd < self.alert_thresholds['drawdown']:
            self._trigger_alert(symbol, 'drawdown', current_dd)
        
        # 2. Volatility spike
        returns = np.diff(prices) / prices[:-1]
        if len(returns) >= 20:
            recent_vol = np.std(returns[-5:])
            normal_vol = np.std(returns[-20:-5])
            
            if recent_vol > normal_vol * self.alert_thresholds['volatility_spike']:
                self._trigger_alert(symbol, 'volatility_spike', recent_vol/normal_vol)
        
        # 3. Volume anomaly
        if len(volumes) >= 20:
            recent_volume = np.mean(volumes[-5:])
            normal_volume = np.mean(volumes[-20:-5])
            
            if recent_volume > normal_volume * self.alert_thresholds['volume_anomaly']:
                self._trigger_alert(symbol, 'volume_anomaly', recent_volume/normal_volume)
    
    def _trigger_alert(self, symbol: str, alert_type: str, value: float):
        """Dispara una alerta"""
        alert = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'type': alert_type,
            'value': value,
            'severity': self._calculate_severity(alert_type, value)
        }
        
        self.metrics_buffer[symbol]['alerts'].append(alert)
        
        # Log alert
        print(f"🚨 ALERT: {symbol} - {alert_type}: {value:.4f} [{alert['severity']}]")
        
        # Aquí se podría integrar con el sistema de notificaciones
        
    def _calculate_severity(self, alert_type: str, value: float) -> str:
        """Calcula severidad de la alerta"""
        if alert_type == 'drawdown':
            if value < -0.10:
                return 'critical'
            elif value < -0.07:
                return 'high'
            else:
                return 'medium'
        
        elif alert_type == 'volatility_spike':
            if value > 3.0:
                return 'high'
            else:
                return 'medium'
                
        elif alert_type == 'volume_anomaly':
            if value > 5.0:
                return 'high'
            else:
                return 'medium'
        
        return 'low'
    
    def get_real_time_summary(self, symbol: str) -> Dict:
        """Obtiene resumen de métricas en tiempo real"""
        if symbol not in self.metrics_buffer:
            return {}
        
        buffer = self.metrics_buffer[symbol]
        prices = np.array(buffer['prices'])
        volumes = np.array(buffer['volumes'])
        
        if len(prices) < 20:
            return {'status': 'insufficient_data'}
        
        # Calcular métricas actuales
        returns = np.diff(prices) / prices[:-1]
        
        summary = {
            'current_price': prices[-1],
            'price_change_1m': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            'price_change_5m': (prices[-1] - prices[-25]) / prices[-25] if len(prices) >= 25 else 0,
            'volatility_1m': np.std(returns[-5:]) if len(returns) >= 5 else 0,
            'volume_ratio': volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1,
            'drawdown': (prices[-1] - np.max(prices)) / np.max(prices),
            'recent_alerts': buffer['alerts'][-5:],  # Últimas 5 alertas
            'momentum': np.mean(returns[-5:]) if len(returns) >= 5 else 0
        }
        
        return summary

# Función principal de inicialización
def create_analytics_system():
    """Crea instancia completa del sistema de analytics"""
    
    system = {
        'portfolio': PortfolioAnalytics(),
        'realtime': RealTimeAnalytics(),
        'attribution': PerformanceAttributionEngine(),
        'risk': AdvancedRiskAnalytics(),
        'ml': MLPerformanceAnalytics()
    }
    
    return system

if __name__ == "__main__":
    # Ejemplo de uso
    analytics = create_analytics_system()
    
    # Simular datos para testing
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    portfolio_returns = pd.Series(
        np.random.normal(0.0005, 0.02, len(dates)),
        index=dates
    )
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.015, len(dates)),
        index=dates
    )
    
    # Generar reporte
    report = analytics['portfolio'].generate_comprehensive_report(
        portfolio_data={'returns': portfolio_returns},
        benchmark_data={'returns': benchmark_returns},
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    print("📊 Analytics Report Generated")
    print(f"Total Return: {report['performance']['portfolio']['total_return']:.2%}")
    print(f"Sharpe Ratio: {report['performance']['portfolio']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['performance']['portfolio']['max_drawdown']:.2%}")