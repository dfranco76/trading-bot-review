# update_ml_labels.py - Actualiza los labels de ML con resultados reales
import sys
sys.path.insert(0, 'src')

from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
import yfinance as yf
from datetime import datetime, timedelta
import json

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class MLLabelUpdater:
    def __init__(self):
        self.supabase = supabase
        print("🤖 ML Label Updater iniciado")
        
    def update_1h_labels(self):
        """Actualiza labels de trades de hace 1 hora"""
        print("\n⏰ Actualizando labels de 1 hora...")
        
        # Obtener features de hace 1 hora sin label de 1h
        one_hour_ago = datetime.now() - timedelta(hours=1, minutes=5)  # 5 min de margen
        
        response = self.supabase.table('ml_features')\
            .select("*")\
            .lt('timestamp', one_hour_ago.isoformat())\
            .is_('actual_price_1h', None)\
            .execute()
        
        features_to_update = response.data
        print(f"  📊 Encontrados {len(features_to_update)} trades para actualizar (1h)")
        
        updated = 0
        for feature in features_to_update:
            try:
                # Obtener precio actual
                symbol = feature['symbol']
                stock = yf.Ticker(symbol)
                current_data = stock.history(period='1d', interval='5m')
                
                if len(current_data) > 0:
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Obtener precio original del JSON
                    features_json = json.loads(feature['features'])
                    original_price = features_json['price']
                    
                    # Calcular cambio
                    price_change_pct = (current_price - original_price) / original_price
                    
                    # Actualizar
                    self.supabase.table('ml_features').update({
                        'actual_price_1h': current_price
                    }).eq('id', feature['id']).execute()
                    
                    updated += 1
                    print(f"  ✅ {symbol}: ${original_price:.2f} → ${current_price:.2f} ({price_change_pct*100:+.2f}%)")
                    
            except Exception as e:
                print(f"  ❌ Error actualizando {feature.get('symbol', 'Unknown')}: {e}")
        
        print(f"  📊 Actualizados {updated} labels de 1h")
        return updated
    
    def update_24h_labels(self):
        """Actualiza labels de trades de hace 24 horas"""
        print("\n📅 Actualizando labels de 24 horas...")
        
        # Obtener features de hace 24 horas sin label completo
        one_day_ago = datetime.now() - timedelta(days=1, minutes=5)
        
        response = self.supabase.table('ml_features')\
            .select("*")\
            .lt('timestamp', one_day_ago.isoformat())\
            .is_('actual_best_action', None)\
            .execute()
        
        features_to_update = response.data
        print(f"  📊 Encontrados {len(features_to_update)} trades para actualizar (24h)")
        
        updated = 0
        for feature in features_to_update:
            try:
                # Obtener precio actual
                symbol = feature['symbol']
                stock = yf.Ticker(symbol)
                current_data = stock.history(period='1d')
                
                if len(current_data) > 0:
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Obtener precio original
                    features_json = json.loads(feature['features'])
                    original_price = features_json['price']
                    
                    # Calcular cambio
                    price_change_pct = (current_price - original_price) / original_price
                    
                    # Determinar mejor acción en retrospectiva
                    if price_change_pct > 0.015:  # Subió más de 1.5%
                        best_action = 'BUY'
                        action_reason = f"Subió {price_change_pct*100:.1f}%"
                    elif price_change_pct < -0.015:  # Bajó más de 1.5%
                        best_action = 'SELL'
                        action_reason = f"Bajó {price_change_pct*100:.1f}%"
                    else:
                        best_action = 'HOLD'
                        action_reason = f"Movimiento lateral {price_change_pct*100:+.1f}%"
                    
                    # Verificar si el sistema tomó la decisión correcta
                    original_prediction = feature['ml_prediction']
                    was_correct = (original_prediction == best_action)
                    
                    # Actualizar
                    self.supabase.table('ml_features').update({
                        'actual_price_24h': current_price,
                        'actual_best_action': best_action
                    }).eq('id', feature['id']).execute()
                    
                    updated += 1
                    emoji = "✅" if was_correct else "❌"
                    print(f"  {emoji} {symbol}: Predijo {original_prediction}, Óptimo era {best_action} ({action_reason})")
                    
            except Exception as e:
                print(f"  ❌ Error actualizando {feature.get('symbol', 'Unknown')}: {e}")
        
        print(f"  📊 Actualizados {updated} labels de 24h")
        return updated
    
    def calculate_model_accuracy(self, days_back=7):
        """Calcula accuracy del modelo actual"""
        print(f"\n📊 Calculando accuracy de los últimos {days_back} días...")
        
        # Obtener features con labels completos
        days_ago = datetime.now() - timedelta(days=days_back)
        
        response = self.supabase.table('ml_features')\
            .select("ml_prediction, actual_best_action")\
            .gt('timestamp', days_ago.isoformat())\
            .not_.is_('actual_best_action', None)\
            .execute()
        
        if not response.data:
            print("  ⚠️ No hay suficientes datos para calcular accuracy")
            return
        
        # Calcular métricas
        total = len(response.data)
        correct = sum(1 for r in response.data if r['ml_prediction'] == r['actual_best_action'])
        accuracy = (correct / total) * 100
        
        # Contar por tipo
        predictions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        actuals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        correct_by_type = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for r in response.data:
            pred = r['ml_prediction']
            actual = r['actual_best_action']
            
            predictions[pred] = predictions.get(pred, 0) + 1
            actuals[actual] = actuals.get(actual, 0) + 1
            
            if pred == actual:
                correct_by_type[pred] = correct_by_type.get(pred, 0) + 1
        
        print(f"\n  📈 RESULTADOS:")
        print(f"  • Accuracy total: {accuracy:.1f}% ({correct}/{total})")
        print(f"\n  • Por tipo de acción:")
        
        for action in ['BUY', 'SELL', 'HOLD']:
            if predictions[action] > 0:
                action_accuracy = (correct_by_type[action] / predictions[action]) * 100
                print(f"    {action}: {action_accuracy:.1f}% ({correct_by_type[action]}/{predictions[action]})")
        
        print(f"\n  • Distribución real optimal:")
        for action in ['BUY', 'SELL', 'HOLD']:
            pct = (actuals[action] / total) * 100
            print(f"    {action}: {pct:.1f}% ({actuals[action]} trades)")
        
        # Guardar en tabla de performance
        self.save_performance_metrics(accuracy, predictions, correct_by_type, total)
        
        return accuracy
    
    def save_performance_metrics(self, accuracy, predictions, correct_by_type, total):
        """Guarda métricas de performance"""
        try:
            # Calcular precision por tipo
            precision_buy = (correct_by_type['BUY'] / predictions['BUY'] * 100) if predictions['BUY'] > 0 else 0
            precision_sell = (correct_by_type['SELL'] / predictions['SELL'] * 100) if predictions['SELL'] > 0 else 0
            
            performance_record = {
                'model_name': 'rule_based_system',
                'model_version': 'v1.0',
                'accuracy': accuracy,
                'precision_buy': precision_buy,
                'precision_sell': precision_sell,
                'training_samples': total,
                'training_date': datetime.now().isoformat(),
                'is_active': True
            }
            
            self.supabase.table('ml_model_performance').insert(performance_record).execute()
            print("\n  💾 Métricas guardadas en ml_model_performance")
            
        except Exception as e:
            print(f"\n  ❌ Error guardando métricas: {e}")
    
    def run_complete_update(self):
        """Ejecuta actualización completa"""
        print("\n🔄 ACTUALIZACIÓN COMPLETA DE LABELS ML")
        print("="*60)
        
        # Actualizar labels de 1h
        updated_1h = self.update_1h_labels()
        
        # Actualizar labels de 24h
        updated_24h = self.update_24h_labels()
        
        # Calcular accuracy si hay suficientes datos
        if updated_24h > 0:
            self.calculate_model_accuracy()
        
        print("\n✅ Actualización completada")
        print(f"📊 Total actualizado: {updated_1h + updated_24h} labels")


def main():
    """Función principal"""
    updater = MLLabelUpdater()
    
    # Si se pasa argumento, ejecutar función específica
    if len(sys.argv) > 1:
        if sys.argv[1] == "1h":
            updater.update_1h_labels()
        elif sys.argv[1] == "24h":
            updater.update_24h_labels()
        elif sys.argv[1] == "accuracy":
            updater.calculate_model_accuracy()
        else:
            print("Uso: python update_ml_labels.py [1h|24h|accuracy]")
    else:
        # Ejecutar actualización completa
        updater.run_complete_update()


if __name__ == "__main__":
    main()