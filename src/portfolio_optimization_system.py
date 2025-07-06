# portfolio_optimization_system.py
"""
Sistema avanzado de optimización de portfolio con teoría moderna
Incluye Black-Litterman, Risk Parity, y optimización robusta
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Intentar importar cvxpy, si no está disponible usar scipy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("cvxpy not available, using scipy optimization instead")

@dataclass
class PortfolioConstraints:
    min_weight: float = 0.0
    max_weight: float = 0.25
    max_positions: int = 20
    max_sector_exposure: float = 0.4
    max_correlation_pairs: float = 0.8
    target_volatility: Optional[float] = None
    min_liquidity: float = 100000  # Volumen diario mínimo

class ModernPortfolioOptimizer:
    """Optimizador de portfolio con métodos avanzados"""
    
    def __init__(self, constraints: PortfolioConstraints = None):
        self.constraints = constraints or PortfolioConstraints()
        self.risk_models = {}
        self.return_models = {}
        
    def optimize_mean_variance(self, returns: pd.DataFrame, 
                             risk_aversion: float = 2.0) -> Dict:
        """Optimización clásica de Markowitz"""
        n_assets = len(returns.columns)
        
        # Calcular estadísticos
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        if CVXPY_AVAILABLE:
            return self._optimize_mean_variance_cvxpy(mean_returns, cov_matrix, risk_aversion)
        else:
            return self._optimize_mean_variance_scipy(mean_returns, cov_matrix, risk_aversion)
    
    def _optimize_mean_variance_cvxpy(self, mean_returns: pd.Series, 
                                     cov_matrix: pd.DataFrame, 
                                     risk_aversion: float) -> Dict:
        """Optimización usando cvxpy"""
        n_assets = len(mean_returns)
        
        # Variables de optimización
        weights = cp.Variable(n_assets)
        
        # Función objetivo: maximizar utilidad (return - risk_aversion * variance)
        portfolio_return = mean_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Restricciones
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.constraints.min_weight,
            weights <= self.constraints.max_weight
        ]
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if weights.value is None:
            return self._equal_weights_fallback(mean_returns.index)
        
        # Calcular métricas
        final_weights = pd.Series(weights.value, index=mean_returns.index)
        portfolio_stats = self._calculate_portfolio_metrics(final_weights, pd.DataFrame(columns=mean_returns.index))
        
        return {
            'weights': final_weights,
            'expected_return': float(mean_returns @ final_weights) * 252,
            'volatility': float(np.sqrt(final_weights @ cov_matrix @ final_weights)) * np.sqrt(252),
            'sharpe_ratio': portfolio_stats.get('sharpe_ratio', 0)
        }
    
    def _optimize_mean_variance_scipy(self, mean_returns: pd.Series, 
                                     cov_matrix: pd.DataFrame, 
                                     risk_aversion: float) -> Dict:
        """Optimización usando scipy (fallback)"""
        n_assets = len(mean_returns)
        
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_variance = weights @ cov_matrix @ weights
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
        
        # Optimizar
        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if not result.success:
            return self._equal_weights_fallback(mean_returns.index)
        
        final_weights = pd.Series(result.x, index=mean_returns.index)
        
        return {
            'weights': final_weights,
            'expected_return': float(mean_returns @ final_weights) * 252,
            'volatility': float(np.sqrt(final_weights @ cov_matrix @ final_weights)) * np.sqrt(252),
            'sharpe_ratio': float(mean_returns @ final_weights) / float(np.sqrt(final_weights @ cov_matrix @ final_weights)) * np.sqrt(252)
        }
    
    def optimize_black_litterman(self, market_caps: pd.Series, 
                               returns: pd.DataFrame,
                               views: Dict[str, float],
                               view_confidences: Dict[str, float]) -> Dict:
        """Modelo Black-Litterman con views del inversor"""
        # Parámetros
        tau = 0.05  # Scalar de incertidumbre
        risk_free_rate = 0.02 / 252  # Diario
        
        # Calcular pesos de mercado
        market_weights = market_caps / market_caps.sum()
        
        # Matriz de covarianza
        cov_matrix = returns.cov()
        
        # Retornos de equilibrio (CAPM inverso)
        risk_aversion = self._estimate_market_risk_aversion(returns, market_weights)
        equilibrium_returns = risk_aversion * cov_matrix @ market_weights
        
        # Construir matriz de views
        P, Q, Omega = self._construct_view_matrices(
            views, view_confidences, returns.columns, cov_matrix
        )
        
        if P.shape[0] == 0:  # No hay views válidas
            return self._optimize_with_custom_returns(
                equilibrium_returns, 
                cov_matrix,
                risk_aversion
            )
        
        # Black-Litterman posterior
        # Σ_posterior = [(τΣ)^-1 + P'Ω^-1P]^-1
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        p_omega_p = P.T @ np.linalg.inv(Omega) @ P
        posterior_cov = np.linalg.inv(tau_sigma_inv + p_omega_p)
        
        # μ_posterior = Σ_posterior[(τΣ)^-1π + P'Ω^-1Q]
        posterior_returns = posterior_cov @ (
            tau_sigma_inv @ equilibrium_returns + 
            P.T @ np.linalg.inv(Omega) @ Q
        )
        
        # Optimizar con retornos posteriores
        return self._optimize_with_custom_returns(
            pd.Series(posterior_returns, index=returns.columns), 
            pd.DataFrame(cov_matrix + posterior_cov, index=returns.columns, columns=returns.columns),
            risk_aversion
        )
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict:
        """Optimización Risk Parity - igual contribución al riesgo"""
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        def risk_contribution(weights):
            """Calcula contribución al riesgo de cada activo"""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol == 0:
                return np.zeros_like(weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            """Minimizar diferencia en contribuciones al riesgo"""
            contrib = risk_contribution(weights)
            # Penalizar desviación de igual contribución
            avg_contrib = np.mean(contrib)
            return np.sum((contrib - avg_contrib) ** 2)
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w - self.constraints.min_weight}
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
        
        # Optimizar
        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-9}
        )
        
        if not result.success:
            return self._equal_weights_fallback(returns.columns)
        
        final_weights = pd.Series(result.x, index=returns.columns)
        portfolio_stats = self._calculate_portfolio_metrics(final_weights, returns)
        
        # Calcular contribuciones al riesgo
        risk_contributions = risk_contribution(result.x)
        
        return {
            'weights': final_weights,
            'risk_contributions': pd.Series(risk_contributions, index=returns.columns),
            'expected_return': portfolio_stats['expected_return'],
            'volatility': portfolio_stats['volatility'],
            'sharpe_ratio': portfolio_stats['sharpe_ratio']
        }
    
    def optimize_maximum_diversification(self, returns: pd.DataFrame) -> Dict:
        """Maximiza ratio de diversificación"""
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        def diversification_ratio(weights):
            """Ratio de diversificación = sum(wi * σi) / σp"""
            weighted_avg_vol = np.sum(weights * volatilities)
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol == 0:
                return 0
            return -weighted_avg_vol / portfolio_vol  # Negativo para maximizar
        
        # Restricciones y bounds
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        bounds = [(0, self.constraints.max_weight)] * n_assets
        
        # Optimizar
        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            diversification_ratio,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        final_weights = pd.Series(result.x, index=returns.columns)
        
        return {
            'weights': final_weights,
            'diversification_ratio': -result.fun,
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def optimize_cvar(self, returns: pd.DataFrame, 
                     confidence_level: float = 0.95) -> Dict:
        """Optimiza Conditional Value at Risk (CVaR)"""
        if CVXPY_AVAILABLE:
            return self._optimize_cvar_cvxpy(returns, confidence_level)
        else:
            return self._optimize_cvar_scipy(returns, confidence_level)
    
    def _optimize_cvar_cvxpy(self, returns: pd.DataFrame, 
                            confidence_level: float) -> Dict:
        """Optimiza CVaR usando cvxpy"""
        n_assets = len(returns.columns)
        n_scenarios = len(returns)
        
        # Variables
        weights = cp.Variable(n_assets)
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Variables auxiliares
        
        # Portfolio returns para cada escenario
        portfolio_returns = returns.values @ weights
        
        # Función objetivo: minimizar CVaR
        alpha = 1 - confidence_level
        cvar = z + (1/alpha) * cp.sum(u) / n_scenarios
        objective = cp.Minimize(cvar)
        
        # Restricciones
        constraints = [
            u >= 0,
            u >= -portfolio_returns - z,
            cp.sum(weights) == 1,
            weights >= self.constraints.min_weight,
            weights <= self.constraints.max_weight
        ]
        
        # Target return constraint (opcional)
        min_return = returns.mean().mean() * 0.8  # 80% del promedio
        constraints.append(returns.values.mean(axis=0) @ weights >= min_return)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if weights.value is None:
            return self._equal_weights_fallback(returns.columns)
        
        final_weights = pd.Series(weights.value, index=returns.columns)
        
        return {
            'weights': final_weights,
            'cvar': float(cvar.value),
            'var': float(z.value),
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def _optimize_cvar_scipy(self, returns: pd.DataFrame, 
                           confidence_level: float) -> Dict:
        """Optimiza CVaR usando scipy (fallback)"""
        n_assets = len(returns.columns)
        n_scenarios = len(returns)
        alpha = 1 - confidence_level
        
        def cvar_objective(x):
            weights = x[:n_assets]
            var = x[n_assets]
            
            portfolio_returns = returns.values @ weights
            cvar = var + np.mean(np.maximum(-portfolio_returns - var, 0)) / alpha
            
            return cvar
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:n_assets]) - 1}
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
        bounds.append((None, None))  # VaR sin límites
        
        # Optimizar
        initial_x = np.concatenate([np.ones(n_assets) / n_assets, [0]])
        result = minimize(
            cvar_objective,
            initial_x,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if not result.success:
            return self._equal_weights_fallback(returns.columns)
        
        final_weights = pd.Series(result.x[:n_assets], index=returns.columns)
        
        return {
            'weights': final_weights,
            'cvar': result.fun,
            'var': result.x[n_assets],
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def optimize_robust(self, returns: pd.DataFrame, 
                       uncertainty_set_size: float = 0.1) -> Dict:
        """Optimización robusta considerando incertidumbre en parámetros"""
        if CVXPY_AVAILABLE:
            return self._optimize_robust_cvxpy(returns, uncertainty_set_size)
        else:
            return self._optimize_robust_scipy(returns, uncertainty_set_size)
    
    def _optimize_robust_cvxpy(self, returns: pd.DataFrame, 
                              uncertainty_set_size: float) -> Dict:
        """Optimización robusta usando cvxpy"""
        n_assets = len(returns.columns)
        
        # Estimaciones puntuales
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Conjuntos de incertidumbre
        return_uncertainty = uncertainty_set_size * np.abs(mean_returns)
        cov_uncertainty = uncertainty_set_size * cov_matrix
        
        # Problema de optimización robusta
        weights = cp.Variable(n_assets)
        
        # Peor caso para retornos (retorno mínimo en conjunto de incertidumbre)
        worst_case_return = mean_returns @ weights - cp.norm(return_uncertainty * weights, 1)
        
        # Peor caso para riesgo (máxima varianza)
        worst_case_variance = cp.quad_form(weights, cov_matrix + cov_uncertainty)
        
        # Función objetivo robusta
        risk_aversion = 2.0
        objective = cp.Maximize(worst_case_return - risk_aversion * worst_case_variance)
        
        # Restricciones
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.constraints.min_weight,
            weights <= self.constraints.max_weight
        ]
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if weights.value is None:
            return self._equal_weights_fallback(returns.columns)
        
        final_weights = pd.Series(weights.value, index=returns.columns)
        
        return {
            'weights': final_weights,
            'worst_case_return': float(worst_case_return.value),
            'worst_case_volatility': float(np.sqrt(worst_case_variance.value)),
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def _optimize_robust_scipy(self, returns: pd.DataFrame, 
                             uncertainty_set_size: float) -> Dict:
        """Optimización robusta usando scipy (fallback)"""
        n_assets = len(returns.columns)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Conjuntos de incertidumbre
        return_uncertainty = uncertainty_set_size * np.abs(mean_returns)
        cov_uncertainty = uncertainty_set_size * cov_matrix
        
        def robust_objective(weights):
            # Peor caso para retornos
            worst_return = mean_returns @ weights - np.sum(return_uncertainty * np.abs(weights))
            
            # Peor caso para varianza
            worst_variance = weights @ (cov_matrix + cov_uncertainty) @ weights
            
            risk_aversion = 2.0
            return -(worst_return - risk_aversion * worst_variance)
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
        
        # Optimizar
        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            robust_objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if not result.success:
            return self._equal_weights_fallback(returns.columns)
        
        final_weights = pd.Series(result.x, index=returns.columns)
        
        # Calcular métricas del peor caso
        worst_return = mean_returns @ result.x - np.sum(return_uncertainty * np.abs(result.x))
        worst_variance = result.x @ (cov_matrix + cov_uncertainty) @ result.x
        
        return {
            'weights': final_weights,
            'worst_case_return': float(worst_return) * 252,
            'worst_case_volatility': float(np.sqrt(worst_variance)) * np.sqrt(252),
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def optimize_hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict:
        """HRP - Hierarchical Risk Parity usando clustering"""
        # Matriz de correlación
        corr_matrix = returns.corr()
        
        # Distancia basada en correlación
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Clustering jerárquico
        from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
        from scipy.spatial.distance import squareform
        
        # Convertir a forma condensada
        condensed_dist = squareform(dist_matrix)
        
        # Clustering
        link = linkage(condensed_dist, method='single')
        
        # Obtener orden de clustering
        root = to_tree(link)
        order = self._get_cluster_order(root, len(returns.columns))
        
        # Reordenar covarianza
        ordered_cov = returns.cov().iloc[order, order]
        
        # Asignación de pesos recursiva
        weights = self._hrp_allocation(ordered_cov)
        
        # Reordenar a índice original
        final_weights = pd.Series(index=returns.columns, dtype=float)
        for i, idx in enumerate(order):
            final_weights.iloc[idx] = weights[i]
        
        return {
            'weights': final_weights,
            'cluster_order': order,
            **self._calculate_portfolio_metrics(final_weights, returns)
        }
    
    def _hrp_allocation(self, cov: pd.DataFrame) -> np.ndarray:
        """Asignación recursiva para HRP"""
        def get_cluster_var(cov, items):
            """Varianza del cluster"""
            cluster_cov = cov.iloc[items, items]
            w = self._get_ivp_weights(cluster_cov)
            return w @ cluster_cov @ w
        
        def recursive_bisection(cov, items):
            """Bisección recursiva"""
            if len(items) == 1:
                return [1.0]
            
            # Dividir en dos clusters
            cluster_1 = items[:len(items)//2]
            cluster_2 = items[len(items)//2:]
            
            # Varianza de cada cluster
            var_1 = get_cluster_var(cov, cluster_1)
            var_2 = get_cluster_var(cov, cluster_2)
            
            # Asignar peso inversamente proporcional a varianza
            alpha = var_2 / (var_1 + var_2) if (var_1 + var_2) > 0 else 0.5
            
            # Recursión
            w_1 = recursive_bisection(cov, cluster_1)
            w_2 = recursive_bisection(cov, cluster_2)
            
            # Combinar pesos
            weights = []
            for w in w_1:
                weights.append(w * alpha)
            for w in w_2:
                weights.append(w * (1 - alpha))
            
            return weights
        
        items = list(range(len(cov)))
        return np.array(recursive_bisection(cov, items))
    
    def _get_ivp_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Inverse Variance Portfolio weights"""
        inv_var = 1 / np.diag(cov)
        return inv_var / inv_var.sum()
    
    def _get_cluster_order(self, root, n):
        """Obtiene orden de elementos en clustering"""
        if root.is_leaf():
            return [root.id]
        else:
            return self._get_cluster_order(root.left, n) + \
                   self._get_cluster_order(root.right, n)
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, 
                                   returns: pd.DataFrame) -> Dict:
        """Calcula métricas del portfolio"""
        # Limpiar weights muy pequeños
        weights = weights[weights > 1e-6]
        if len(weights) == 0:
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'effective_assets': 0
            }
        
        weights = weights / weights.sum()
        
        # Si no hay suficientes datos de returns, devolver métricas básicas
        if len(returns) < 2:
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'effective_assets': (weights ** 2).sum() ** -1
            }
        
        # Returns del portfolio
        portfolio_returns = returns[weights.index] @ weights
        
        # Métricas
        expected_return = portfolio_returns.mean() * 252  # Anualizado
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = expected_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'effective_assets': (weights ** 2).sum() ** -1  # Número efectivo de activos
        }
    
    def _estimate_market_risk_aversion(self, returns: pd.DataFrame, 
                                     market_weights: pd.Series) -> float:
        """Estima aversión al riesgo del mercado"""
        # Alinear índices
        common_assets = returns.columns.intersection(market_weights.index)
        aligned_weights = market_weights[common_assets]
        aligned_weights = aligned_weights / aligned_weights.sum()
        
        market_return = (returns[common_assets] @ aligned_weights).mean() * 252
        market_vol = (returns[common_assets] @ aligned_weights).std() * np.sqrt(252)
        risk_free = 0.02  # 2% anual
        
        # Desde CAPM: E[Rm] - Rf = λ * σm²
        if market_vol > 0:
            return (market_return - risk_free) / (market_vol ** 2)
        else:
            return 2.0  # Valor por defecto
    
    def _construct_view_matrices(self, views: Dict[str, float], 
                               confidences: Dict[str, float],
                               assets: pd.Index, 
                               cov_matrix: pd.DataFrame) -> Tuple:
        """Construye matrices P, Q, Omega para Black-Litterman"""
        # Filtrar views para activos existentes
        valid_views = {k: v for k, v in views.items() if k in assets}
        n_views = len(valid_views)
        
        if n_views == 0:
            return np.array([]), np.array([]), np.array([[]])
        
        n_assets = len(assets)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)
        
        for i, (asset, view_return) in enumerate(valid_views.items()):
            asset_idx = assets.get_loc(asset)
            P[i, asset_idx] = 1
            Q[i] = view_return
            
            # Incertidumbre de la view
            confidence = confidences.get(asset, 0.5)
            view_variance = cov_matrix.iloc[asset_idx, asset_idx]
            omega_diag[i] = view_variance / confidence
        
        Omega = np.diag(omega_diag)
        
        return P, Q, Omega
    
    def _optimize_with_custom_returns(self, returns: pd.Series, 
                                    cov_matrix: pd.DataFrame,
                                    risk_aversion: float) -> Dict:
        """Optimización con retornos personalizados"""
        n_assets = len(returns)
        
        if CVXPY_AVAILABLE:
            # Variables de optimización
            weights = cp.Variable(n_assets)
            
            # Función objetivo
            portfolio_return = returns.values @ weights
            portfolio_variance = cp.quad_form(weights, cov_matrix.values)
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
            
            # Restricciones
            constraints = [
                cp.sum(weights) == 1,
                weights >= self.constraints.min_weight,
                weights <= self.constraints.max_weight
            ]
            
            # Resolver
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if weights.value is None:
                return self._equal_weights_fallback(returns.index)
            
            final_weights = pd.Series(weights.value, index=returns.index)
        else:
            # Usar scipy
            def objective(weights):
                portfolio_return = np.sum(returns * weights)
                portfolio_variance = weights @ cov_matrix @ weights
                return -(portfolio_return - risk_aversion * portfolio_variance)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
            
            initial_weights = np.ones(n_assets) / n_assets
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds
            )
            
            if not result.success:
                return self._equal_weights_fallback(returns.index)
            
            final_weights = pd.Series(result.x, index=returns.index)
        
        # Calcular métricas
        expected_return = float(returns @ final_weights) * 252
        volatility = float(np.sqrt(final_weights @ cov_matrix @ final_weights)) * np.sqrt(252)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            'weights': final_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _equal_weights_fallback(self, assets: pd.Index) -> Dict:
        """Fallback a pesos iguales"""
        n = len(assets)
        weights = pd.Series(1/n, index=assets)
        
        return {
            'weights': weights,
            'expected_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'method': 'equal_weights_fallback'
        }

class DynamicPortfolioRebalancer:
    """Sistema de rebalanceo dinámico"""
    
    def __init__(self, optimizer: ModernPortfolioOptimizer):
        self.optimizer = optimizer
        self.rebalance_history = []
        self.transaction_costs = 0.001  # 10 bps
        
    def should_rebalance(self, current_weights: pd.Series, 
                        target_weights: pd.Series,
                        current_prices: pd.Series,
                        threshold: float = 0.05) -> Tuple[bool, Dict]:
        """Determina si se debe rebalancear"""
        # Alinear índices
        common_assets = current_weights.index.intersection(target_weights.index)
        current_aligned = current_weights[common_assets]
        target_aligned = target_weights[common_assets]
        
        # Normalizar pesos
        current_aligned = current_aligned / current_aligned.sum()
        target_aligned = target_aligned / target_aligned.sum()
        
        # Calcular drift
        weight_drift = np.abs(current_aligned - target_aligned)
        max_drift = weight_drift.max()
        
        # Calcular costo de rebalanceo
        turnover = weight_drift.sum()
        rebalance_cost = turnover * self.transaction_costs
        
        # Beneficio esperado del rebalanceo
        # (simplificado - en producción usar modelo más sofisticado)
        expected_benefit = max_drift * 0.02  # 2% anual por 5% drift
        
        should_rebalance = expected_benefit > rebalance_cost * 2  # Factor de seguridad
        
        return should_rebalance, {
            'max_drift': max_drift,
            'total_drift': weight_drift.sum(),
            'rebalance_cost': rebalance_cost,
            'expected_benefit': expected_benefit,
            'cost_benefit_ratio': rebalance_cost / expected_benefit if expected_benefit > 0 else float('inf')
        }
    
    def calculate_rebalance_trades(self, current_positions: Dict[str, float],
                                 target_weights: pd.Series,
                                 total_value: float) -> List[Dict]:
        """Calcula trades necesarios para rebalancear"""
        trades = []
        
        # Calcular posiciones objetivo
        target_values = target_weights * total_value
        
        # Calcular diferencias
        for asset, target_value in target_values.items():
            current_value = current_positions.get(asset, 0)
            diff_value = target_value - current_value
            
            if abs(diff_value) > total_value * 0.001:  # Mínimo 0.1%
                trades.append({
                    'symbol': asset,
                    'action': 'BUY' if diff_value > 0 else 'SELL',
                    'value': abs(diff_value),
                    'current_weight': current_value / total_value if total_value > 0 else 0,
                    'target_weight': target_weights[asset]
                })
        
        # Ordenar por tamaño para ejecutar grandes primero
        trades.sort(key=lambda x: x['value'], reverse=True)
        
        return trades

class FactorPortfolioOptimizer:
    """Optimización basada en factores"""
    
    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_models = {}
        
    def optimize_factor_portfolio(self, factor_exposures: pd.DataFrame,
                                factor_returns: pd.DataFrame,
                                factor_covariance: pd.DataFrame,
                                target_exposures: Dict[str, float] = None) -> Dict:
        """Optimiza portfolio basado en exposición a factores"""
        n_assets = len(factor_exposures)
        n_factors = len(self.factors)
        
        if CVXPY_AVAILABLE:
            # Variables
            weights = cp.Variable(n_assets)
            
            # Exposiciones del portfolio a factores
            portfolio_exposures = factor_exposures.T @ weights
            
            # Retorno esperado basado en factores
            expected_factor_returns = factor_returns.mean()
            portfolio_return = expected_factor_returns @ portfolio_exposures
            
            # Riesgo basado en factores
            portfolio_factor_risk = cp.quad_form(portfolio_exposures, factor_covariance.values)
            
            # Función objetivo
            risk_aversion = 2.0
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_factor_risk)
            
            # Restricciones
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= 0.1  # Max 10% por activo
            ]
            
            # Restricciones de exposición a factores si se especifican
            if target_exposures:
                for factor, target in target_exposures.items():
                    if factor in factor_exposures.columns:
                        factor_idx = factor_exposures.columns.get_loc(factor)
                        constraints.append(
                            cp.abs(portfolio_exposures[factor_idx] - target) <= 0.1
                        )
            
            # Resolver
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if weights.value is None:
                return self._equal_weights_fallback(factor_exposures.index)
            
            final_weights = pd.Series(weights.value, index=factor_exposures.index)
            final_exposures = factor_exposures.T @ final_weights
            
            return {
                'weights': final_weights,
                'factor_exposures': final_exposures,
                'expected_return': float(expected_factor_returns @ final_exposures) * 252,
                'factor_risk': float(np.sqrt(final_exposures @ factor_covariance @ final_exposures)) * np.sqrt(252)
            }
        else:
            # Fallback sin cvxpy
            return self._optimize_factor_scipy(
                factor_exposures, factor_returns, factor_covariance, target_exposures
            )
    
    def _optimize_factor_scipy(self, factor_exposures: pd.DataFrame,
                             factor_returns: pd.DataFrame,
                             factor_covariance: pd.DataFrame,
                             target_exposures: Dict[str, float] = None) -> Dict:
        """Optimización de factores usando scipy"""
        n_assets = len(factor_exposures)
        expected_factor_returns = factor_returns.mean()
        
        def objective(weights):
            portfolio_exposures = factor_exposures.T @ weights
            portfolio_return = expected_factor_returns @ portfolio_exposures
            portfolio_risk = portfolio_exposures @ factor_covariance @ portfolio_exposures
            risk_aversion = 2.0
            return -(portfolio_return - risk_aversion * portfolio_risk)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Agregar restricciones de exposición si existen
        if target_exposures:
            for factor, target in target_exposures.items():
                if factor in factor_exposures.columns:
                    factor_idx = factor_exposures.columns.get_loc(factor)
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=factor_idx, t=target: 0.1 - abs((factor_exposures.T @ w)[idx] - t)
                    })
        
        bounds = [(0, 0.1)] * n_assets
        
        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if not result.success:
            return self._equal_weights_fallback(factor_exposures.index)
        
        final_weights = pd.Series(result.x, index=factor_exposures.index)
        final_exposures = factor_exposures.T @ final_weights
        
        return {
            'weights': final_weights,
            'factor_exposures': final_exposures,
            'expected_return': float(expected_factor_returns @ final_exposures) * 252,
            'factor_risk': float(np.sqrt(final_exposures @ factor_covariance @ final_exposures)) * np.sqrt(252)
        }
    
    def _equal_weights_fallback(self, assets: pd.Index) -> Dict:
        """Fallback a pesos iguales"""
        n = len(assets)
        weights = pd.Series(1/n, index=assets)
        
        return {
            'weights': weights,
            'factor_exposures': pd.Series(0, index=self.factors),
            'expected_return': 0,
            'factor_risk': 0
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    # Simular retornos
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.random.uniform(0.0001, 0.001, n_assets),
            cov=np.random.randn(n_assets, n_assets) @ np.random.randn(n_assets, n_assets).T * 0.0001,
            size=n_days
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Crear optimizador
    optimizer = ModernPortfolioOptimizer()
    
    # Optimización Mean-Variance
    print("Mean-Variance Optimization:")
    mv_result = optimizer.optimize_mean_variance(returns, risk_aversion=2.0)
    print(f"Expected Return: {mv_result['expected_return']:.2%}")
    print(f"Volatility: {mv_result['volatility']:.2%}")
    print(f"Sharpe Ratio: {mv_result['sharpe_ratio']:.2f}")
    print()
    
    # Risk Parity
    print("Risk Parity Optimization:")
    rp_result = optimizer.optimize_risk_parity(returns)
    print(f"Expected Return: {rp_result['expected_return']:.2%}")
    print(f"Volatility: {rp_result['volatility']:.2%}")
    print(f"Risk Contributions std: {rp_result['risk_contributions'].std():.4f}")
    print()
    
    # Maximum Diversification
    print("Maximum Diversification:")
    md_result = optimizer.optimize_maximum_diversification(returns)
    print(f"Diversification Ratio: {md_result['diversification_ratio']:.2f}")
    print(f"Expected Return: {md_result['expected_return']:.2%}")
    print(f"Volatility: {md_result['volatility']:.2%}")