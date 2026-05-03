import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
import os

from app.api.core.config import settings
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3

# ==========================================
# 1. SUAS CHAVES GRATUITAS AQUI
# ==========================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Inicializa o 'motor' oficial da Hugging Face
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

def get_min_date_from_s3(ticker: str) -> int:
    """
    Lê o CSV Master gerado pelo pipeline do S3 para descobrir a data mínima já processada.
    """
    bucket = settings.S3_BUCKET_NAME
    s3_key = f"data/historical/{ticker}_master_history.csv"
    
    df = read_csv_from_s3(bucket, s3_key)
    
    if df is None or df.empty:
        print(f"❌ Arquivo master não encontrado no S3: {s3_key} \n Retornando 360 dias (Carga Total)")
        return 360
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        
        hoje = pd.Timestamp.now().normalize()
        min_date = min_date.normalize()
        dias_passado = (hoje - min_date).days
        
        print(f"📅 Data mínima encontrada no S3: {min_date.date()} \n Gerando notícias dos últimos {dias_passado} dias")
        return dias_passado
    except Exception as e:
        print(f"Erro ao processar datas do CSV: {e} \n Retornando 360 dias")
        return 360

def analisar_sentimento_finbert(texto, max_tentativas=3):
    """Manda a manchete para a nuvem usando o SDK Oficial com Tolerância a Falhas (Retry)"""
    for tentativa in range(max_tentativas):
        try:
            resultados = hf_client.text_classification(texto, model="ProsusAI/finbert")
            
            score_final = 0
            for item in resultados:
                label = item['label'].lower() if isinstance(item, dict) else item.label.lower()
                score = item['score'] if isinstance(item, dict) else item.score
                
                if label == 'positive':
                    score_final += score
                elif label == 'negative':
                    score_final -= score
                    
            return score_final
            
        except Exception as e:
            erro_str = str(e)
            if "500" in erro_str or "503" in erro_str or "504" in erro_str:
                print(f"   ⚠️ Servidor HF engasgou. Tentativa {tentativa+1}/{max_tentativas}. Aguardando 3s...")
                time.sleep(3)
            else:
                print(f"❌ Erro fatal na IA da Hugging Face: {e}")
                return 0
                
    return 0

def gerar_base_sentimento(ticker="RACE", dias_passado=30):
    """Baixa notícias do Finnhub, pontua no FinBERT e salva o consolidado no S3"""
    
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias_passado)
    
    str_inicio = data_inicio.strftime('%Y-%m-%d')
    str_fim = data_fim.strftime('%Y-%m-%d')
    
    print(f"📥 Buscando notícias de {ticker} ({str_inicio} até {str_fim}) no Finnhub...")
    url_news = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={str_inicio}&to={str_fim}&token={FINNHUB_API_KEY}"
    
    response_news = requests.get(url_news)
    noticias = response_news.json()
    
    if isinstance(noticias, dict) and "error" in noticias:
        print(f"❌ API do Finnhub bloqueou a requisição. Motivo: {noticias['error']}")
        return
        
    if not isinstance(noticias, list) or len(noticias) == 0:
        print("❌ Nenhuma notícia encontrada para este período.")
        return
        
    print(f"✅ {len(noticias)} manchetes encontradas! Analisando sentimento na Nuvem...")
    
    dados_processados = []
    
    for i, noticia in enumerate(noticias):
        manchete = noticia.get('headline', '')
        if not manchete:
            continue
            
        data_publicacao = datetime.fromtimestamp(noticia['datetime']).strftime('%Y-%m-%d')
        
        if i % 10 == 0:
            print(f"Processando {i}/{len(noticias)}...")
            
        score = analisar_sentimento_finbert(manchete)
        
        dados_processados.append({
            "Date": data_publicacao,
            "Headline": manchete,
            "Sentiment_Score": score
        })
        
        time.sleep(1)

    df_noticias = pd.DataFrame(dados_processados)
    
    if df_noticias.empty:
        print("❌ Nenhuma notícia pôde ser processada.")
        return

    # Agrupa tirando a MÉDIA do sentimento do dia
    df_diario_novo = df_noticias.groupby("Date")["Sentiment_Score"].mean().reset_index()
    
    # ==========================================
    # LÓGICA DE MLOps: Carga Incremental no S3
    # ==========================================
    bucket = settings.S3_BUCKET_NAME
    s3_key = f"data/features/{ticker}_news_sentiment.csv"
    
    # Tenta baixar o histórico antigo para concatenar
    df_antigo = read_csv_from_s3(bucket, s3_key)
    
    if df_antigo is not None and not df_antigo.empty:
        print("🔄 Mesclando novas notícias com o histórico do S3...")
        df_consolidado = pd.concat([df_antigo, df_diario_novo])
        # Garante que não teremos duas médias para o mesmo dia
        df_consolidado = df_consolidado.drop_duplicates(subset=['Date'], keep='last')
        df_consolidado = df_consolidado.sort_values(by='Date')
    else:
        df_consolidado = df_diario_novo
        print("🆕 Criando novo arquivo base de sentimento no S3.")

    # Salva de volta na nuvem
    write_csv_to_s3(bucket, s3_key, df_consolidado)
    print(f"🚀 Sucesso! Arquivo '{s3_key}' salvo no Data Lake com sentimento matemático limpo.")

if __name__ == "__main__":
    ticker_alvo = "RACE"
    dias = get_min_date_from_s3(ticker_alvo)
    gerar_base_sentimento(ticker_alvo, dias_passado=dias)