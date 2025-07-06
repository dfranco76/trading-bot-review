# src/agente_sentiment.py
from config import SUPABASE_URL, SUPABASE_KEY, NEWSAPI_KEY
from supabase import create_client, Client
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import requests
from textblob import TextBlob
import re
import json

# Configuración Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AgenteSentiment:
    def __init__(self):
        self.nombre = "Agente Sentiment"
        self.capital = self.get_capital()
        print(f"✅ {self.nombre} iniciado con ${self.capital}")
        
        # Verificar si NewsAPI está configurada
        self.newsapi_configured = bool(NEWSAPI_KEY and NEWSAPI_KEY != "" and NEWSAPI_KEY != "tu_newsapi_key")
        if self.newsapi_configured:
            print(f"  📰 NewsAPI configurada correctamente")
        else:
            print(f"  ⚠️ NewsAPI no configurada - usando solo Yahoo Finance")
        
    def get_capital(self):
        """Obtiene capital actual de la BD"""
        try:
            response = supabase.table('bot_status').select("capital").execute()
            return float(response.data[0]['capital']) if response.data else 200.0
        except:
            return 200.0
    
    def get_newsapi_sentiment(self, symbol):
        """Obtiene sentiment de NewsAPI"""
        if not self.newsapi_configured:
            return 0, []
        
        try:
            # Mapeo de símbolos a nombres de compañías para mejor búsqueda
            company_names = {
                'NVDA': 'NVIDIA',
                'TSLA': 'Tesla',
                'PLTR': 'Palantir',
                'SOFI': 'SoFi',
                'NET': 'Cloudflare',
                'AMD': 'AMD',
                'COIN': 'Coinbase',
                'MARA': 'Marathon Digital',
                'SQ': 'Block Square',
                'HOOD': 'Robinhood',
                'DDOG': 'Datadog',
                'SNOW': 'Snowflake'
            }
            
            company_name = company_names.get(symbol, symbol)
            
            # Construir query para NewsAPI
            query = f'"{company_name}" OR "{symbol}" stock'
            
            # Fecha de hace 7 días
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Llamar a NewsAPI
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 20,
                'apiKey': NEWSAPI_KEY
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                if not articles:
                    return 0, []
                
                sentiments = []
                headlines = []
                
                for article in articles[:10]:  # Analizar máximo 10 artículos
                    # Combinar título y descripción para análisis
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    headlines.append(article.get('title', ''))
                    
                    # Análisis de sentiment con TextBlob
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    
                    # Ajustar por palabras clave financieras
                    positive_keywords = ['upgrade', 'beat', 'surge', 'rally', 'gain', 'profit', 'growth', 'bullish', 'buy']
                    negative_keywords = ['downgrade', 'miss', 'fall', 'drop', 'loss', 'bear', 'sell', 'crash', 'concern']
                    
                    text_lower = text.lower()
                    for word in positive_keywords:
                        if word in text_lower:
                            sentiment_score += 0.1
                    
                    for word in negative_keywords:
                        if word in text_lower:
                            sentiment_score -= 0.1
                    
                    # Limitar entre -1 y 1
                    sentiment_score = max(-1, min(1, sentiment_score))
                    sentiments.append(sentiment_score)
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                return avg_sentiment, headlines[:5]  # Retornar top 5 titulares
            
            else:
                # Error de API
                if response.status_code == 429:
                    print(f"  ⚠️ Límite de NewsAPI alcanzado")
                return 0, []
                
        except Exception as e:
            print(f"  ❌ Error con NewsAPI: {e}")
            return 0, []
    
    def get_yahoo_sentiment(self, symbol):
        """Obtiene sentiment de Yahoo Finance (backup)"""
        try:
            stock = yf.Ticker(symbol)
            
            # Obtener noticias recientes
            news = stock.news
            if not news:
                return 0, []
            
            sentiments = []
            headlines = []
            
            for article in news[:10]:  # Últimas 10 noticias
                title = article.get('title', '')
                headlines.append(title)
                
                # Análisis simple de sentiment
                blob = TextBlob(title)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            return avg_sentiment, headlines[:5]
            
        except:
            return 0, []
    
    def analyze_price_sentiment(self, data):
        """Analiza sentiment basado en acción del precio"""
        if len(data) < 5:
            return 0
        
        # Calcular cambios diarios
        daily_changes = data['Close'].pct_change().dropna()
        
        # Sentiment metrics
        positive_days = sum(daily_changes > 0)
        negative_days = sum(daily_changes < 0)
        avg_change = daily_changes.mean()
        
        # Momentum a corto plazo (últimos 5 días)
        recent_momentum = daily_changes[-5:].mean()
        
        # Price sentiment score
        price_sentiment = 0
        
        # Tendencia general
        if positive_days > negative_days * 1.5:
            price_sentiment = 0.5  # Bullish
        elif negative_days > positive_days * 1.5:
            price_sentiment = -0.5  # Bearish
        
        # Ajustar por momentum reciente
        if recent_momentum > 0.02:  # +2% promedio últimos 5 días
            price_sentiment += 0.3
        elif recent_momentum < -0.02:  # -2% promedio
            price_sentiment -= 0.3
        
        # Ajustar por magnitud
        if abs(avg_change) > 0.03:  # Movimientos fuertes
            price_sentiment *= 1.5
        
        return np.clip(price_sentiment, -1, 1)
    
    def calculate_social_volume(self, symbol):
        """Calcula volumen social basado en volumen de trading"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="5d")
            
            if len(data) < 2:
                return 0
            
            # Comparar volumen actual vs promedio
            current_vol = data['Volume'].iloc[-1]
            avg_vol = data['Volume'].mean()
            
            volume_spike = (current_vol / avg_vol) - 1 if avg_vol > 0 else 0
            
            # Interpretación
            if volume_spike > 1.0:  # Volumen 2x o más
                return min(volume_spike / 2, 1.0)  # Normalizar
            elif volume_spike > 0.5:  # Volumen 50% más alto
                return volume_spike
            elif volume_spike < -0.3:  # Volumen 30% más bajo
                return volume_spike
            else:
                return 0
                
        except:
            return 0
    
    def detect_fear_greed(self, data, rsi):
        """Detecta niveles de miedo o codicia"""
        if len(data) < 20:
            return 0
        
        # Volatilidad (VIX proxy)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Anualizada
        
        # Fear & Greed indicators
        fear_greed = 0
        
        # RSI extremos
        if rsi > 80:
            fear_greed = 0.8  # Codicia extrema
        elif rsi > 70:
            fear_greed = 0.5  # Codicia
        elif rsi < 20:
            fear_greed = -0.8  # Miedo extremo
        elif rsi < 30:
            fear_greed = -0.5  # Miedo
        
        # Ajustar por volatilidad
        if volatility > 0.4:  # Alta volatilidad = más miedo
            fear_greed -= 0.3
        elif volatility < 0.15:  # Baja volatilidad = complacencia
            fear_greed += 0.2
        
        return np.clip(fear_greed, -1, 1)
    
    def analyze_combined_sentiment(self, news_sentiment, price_sentiment, social_volume, fear_greed):
        """Combina todos los tipos de sentiment con pesos dinámicos"""
        # Pesos base
        weights = {
            'news': 0.35 if self.newsapi_configured else 0.25,
            'price': 0.30,
            'social': 0.20,
            'fear_greed': 0.15 if self.newsapi_configured else 0.25
        }
        
        # Ajustar pesos si alguna señal es muy fuerte
        if abs(news_sentiment) > 0.7:
            weights['news'] *= 1.3
        if abs(price_sentiment) > 0.7:
            weights['price'] *= 1.2
        if abs(social_volume) > 0.7:
            weights['social'] *= 1.2
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        # Calcular sentiment combinado
        combined = (
            news_sentiment * weights['news'] +
            price_sentiment * weights['price'] +
            social_volume * weights['social'] +
            fear_greed * weights['fear_greed']
        )
        
        return combined, weights
    
    def analyze_symbol(self, symbol):
        """Análisis de sentiment completo mejorado"""
        try:
            # Obtener datos
            stock = yf.Ticker(symbol)
            data = stock.history(period="1mo")
            
            if len(data) < 20:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'Datos insuficientes para sentiment'
                }
            
            precio_actual = data['Close'].iloc[-1]
            
            # Calcular RSI para fear/greed
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
            
            # Obtener diferentes tipos de sentiment
            # Primero intentar NewsAPI, luego Yahoo como backup
            if self.newsapi_configured:
                news_sentiment, headlines = self.get_newsapi_sentiment(symbol)
                if not headlines:  # Si NewsAPI no retornó nada, usar Yahoo
                    news_sentiment, headlines = self.get_yahoo_sentiment(symbol)
            else:
                news_sentiment, headlines = self.get_yahoo_sentiment(symbol)
            
            price_sentiment = self.analyze_price_sentiment(data)
            social_volume = self.calculate_social_volume(symbol)
            fear_greed = self.detect_fear_greed(data, rsi)
            
            # Combinar sentiments
            sentiment_score, weights_used = self.analyze_combined_sentiment(
                news_sentiment, price_sentiment, social_volume, fear_greed
            )
            
            print(f"  😊 {symbol}: Sentiment Analysis")
            print(f"  📰 News sentiment: {news_sentiment:.2f} (peso: {weights_used['news']:.0%})")
            print(f"  📈 Price sentiment: {price_sentiment:.2f} (peso: {weights_used['price']:.0%})")
            print(f"  🔊 Social volume: {social_volume:.2f} (peso: {weights_used['social']:.0%})")
            print(f"  😨 Fear/Greed: {fear_greed:.2f} (RSI: {rsi:.0f}) (peso: {weights_used['fear_greed']:.0%})")
            print(f"  📊 Overall sentiment: {sentiment_score:.2f}")
            
            if headlines:
                print(f"  📰 Últimas noticias:")
                for h in headlines[:3]:
                    print(f"     • {h[:60]}...")
            
            # ESTRATEGIAS DE SENTIMENT MEJORADAS
            
            # 1. Pánico extremo = Oportunidad de compra
            if fear_greed < -0.7 and price_sentiment < -0.3:
                return {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'price': precio_actual,
                    'reason': f'Pánico extremo detectado (Fear: {fear_greed:.2f})'
                }
            
            # 2. Sentiment muy positivo con momentum
            elif sentiment_score > 0.6 and price_sentiment > 0.3:
                return {
                    'action': 'BUY',
                    'confidence': min(0.8, sentiment_score),
                    'price': precio_actual,
                    'reason': f'Sentiment muy positivo ({sentiment_score:.2f})'
                }
            
            # 3. Divergencia: noticias muy positivas pero precio no sube
            elif news_sentiment > 0.5 and price_sentiment < 0:
                return {
                    'action': 'BUY',
                    'confidence': 0.65,
                    'price': precio_actual,
                    'reason': f'Divergencia positiva: buenas noticias ({news_sentiment:.2f}) con precio rezagado'
                }
            
            # 4. Divergencia negativa: precio sube pero sentiment negativo
            elif price_sentiment > 0.3 and news_sentiment < -0.3:
                return {
                    'action': 'SELL',
                    'confidence': 0.70,
                    'price': precio_actual,
                    'reason': 'Divergencia: precio sube pero noticias negativas'
                }
            
            # 5. Codicia extrema con volumen alto
            elif fear_greed > 0.7 and social_volume > 0.5:
                return {
                    'action': 'SELL',
                    'confidence': 0.75,
                    'price': precio_actual,
                    'reason': f'Codicia extrema + alto volumen social'
                }
            
            # 6. Sentiment negativo fuerte
            elif sentiment_score < -0.6:
                return {
                    'action': 'SELL',
                    'confidence': min(0.75, abs(sentiment_score)),
                    'price': precio_actual,
                    'reason': f'Sentiment muy negativo ({sentiment_score:.2f})'
                }
            
            # 7. Momentum de noticias (muchas noticias positivas recientes)
            elif news_sentiment > 0.3 and len(headlines) >= 5 and social_volume > 0.3:
                return {
                    'action': 'BUY',
                    'confidence': 0.60,
                    'price': precio_actual,
                    'reason': f'Momentum de noticias positivas ({len(headlines)} artículos)'
                }
            
            # 8. Sentiment moderadamente positivo
            elif sentiment_score > 0.3:
                return {
                    'action': 'BUY',
                    'confidence': 0.55,
                    'price': precio_actual,
                    'reason': f'Sentiment moderadamente positivo ({sentiment_score:.2f})'
                }
            
            # 9. Sentiment moderadamente negativo
            elif sentiment_score < -0.3:
                return {
                    'action': 'SELL',
                    'confidence': 0.55,
                    'price': precio_actual,
                    'reason': f'Sentiment moderadamente negativo ({sentiment_score:.2f})'
                }
            
            # Sin señal clara
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': precio_actual,
                'reason': f'Sentiment neutral ({sentiment_score:.2f})'
            }
            
        except Exception as e:
            print(f"  ❌ Error analizando sentiment {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': f'Error en análisis: {str(e)}'
            }
    
    def execute_trade(self, symbol, signal):
        """Registra trade en la BD"""
        if signal['action'] in ['BUY', 'SELL']:
            try:
                # Calcular cantidad
                cantidad = (self.capital * 0.15) / signal['price']
                
                # Guardar en BD
                trade = {
                    'symbol': symbol,
                    'action': signal['action'],
                    'quantity': cantidad,
                    'price': signal['price'],
                    'agent_confidence': signal['confidence'],
                    'agent_name': self.nombre,
                    'strategy_reason': signal['reason']
                }
                
                supabase.table('trades').insert(trade).execute()
                
                emoji = "😊" if signal['action'] == 'BUY' else "😰"
                print(f"{emoji} {signal['action']} (Sentiment): {cantidad:.2f} {symbol} @ ${signal['price']:.2f}")
                print(f"   Razón: {signal['reason']}")
                print(f"   Confianza: {signal['confidence']*100:.0f}%")
            except Exception as e:
                print(f"❌ Error ejecutando trade: {e}")
    
    def run_cycle(self, symbols):
        """Ejecuta un ciclo de análisis"""
        print(f"\n⏰ Ciclo Sentiment Analysis - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        encontro_algo = False
        for symbol in symbols:
            print(f"\nAnalizando sentiment de {symbol}...")
            signal = self.analyze_symbol(symbol)
            
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                encontro_algo = True
        
        if not encontro_algo:
            print("\n🔍 No encontré señales de sentiment claras")
        
        print("\n✅ Ciclo completado")


if __name__ == "__main__":
    print("😊 SENTIMENT TRADING AGENT")
    print("Analizando emociones del mercado y noticias")
    print("="*50)
    
    # Crear agente
    agente = AgenteSentiment()
    
    # Símbolos a analizar
    from config import SYMBOLS
    
    # Ejecutar análisis
    agente.run_cycle(SYMBOLS)