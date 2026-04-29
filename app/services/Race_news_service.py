import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
import os


# ==========================================
# 1. SUAS CHAVES GRATUITAS AQUI
# ==========================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Inicializa o 'motor' oficial da Hugging Face
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

def get_min_date_from_csv(caminho_csv) -> int:
    """Lê o CSV gerado pelo pipeline de features para descobrir a data mínima já processada"""
    if not os.path.exists(caminho_csv):
        print(f"❌ Arquivo CSV não encontrado: {caminho_csv} \n retornando 360 dias")
        return 360
    
    try:
        df = pd.read_csv(caminho_csv)
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        
        # CORREÇÃO: Usando Pandas Timestamp para calcular corretamente (Hoje - Passado)
        hoje = pd.Timestamp.now().normalize()
        min_date = min_date.normalize()
        dias_passado = (hoje - min_date).days
        
        print(f"📅 Data mínima encontrada no CSV: {min_date.date()} \n Gerando notícias dos últimos {dias_passado} dias")
        return dias_passado
    except Exception as e:
        print(f"Erro ao ler o CSV: {e} \n retornando 360 dias")
        return 360

def analisar_sentimento_finbert(texto, max_tentativas=3):
    """Manda a manchete para a nuvem usando o SDK Oficial com Tolerância a Falhas (Retry)"""
    for tentativa in range(max_tentativas):
        try:
            # A biblioteca cuida de achar a URL certa, passar pela segurança e acordar o modelo
            resultados = hf_client.text_classification(texto, model="ProsusAI/finbert")
            
            score_final = 0
            for item in resultados:
                # A biblioteca pode retornar objetos ou dicionários, cobrimos os dois casos
                label = item['label'].lower() if isinstance(item, dict) else item.label.lower()
                score = item['score'] if isinstance(item, dict) else item.score
                
                if label == 'positive':
                    score_final += score
                elif label == 'negative':
                    score_final -= score
                    
            return score_final
            
        except Exception as e:
            erro_str = str(e)
            # Se for erro 500, 503 ou 504 (Instabilidade na HF), dorme 3s e tenta de novo
            if "500" in erro_str or "503" in erro_str or "504" in erro_str:
                print(f"   ⚠️ Servidor HF engasgou. Tentativa {tentativa+1}/{max_tentativas}. Aguardando 3s...")
                time.sleep(3)
            else:
                print(f"❌ Erro fatal na IA da Hugging Face: {e}")
                return 0
                
    # Se bater as 3 tentativas e falhar, assume sentimento neutro para não quebrar o script
    return 0

def gerar_base_sentimento(ticker="RACE", dias_passado=30):
    """Baixa notícias do Finnhub, pontua no FinBERT e gera um CSV de médias diárias"""
    
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias_passado)
    
    str_inicio = data_inicio.strftime('%Y-%m-%d')
    str_fim = data_fim.strftime('%Y-%m-%d')
    
    print(f"📥 Buscando notícias de {ticker} ({str_inicio} até {str_fim}) no Finnhub...")
    url_news = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={str_inicio}&to={str_fim}&token={FINNHUB_API_KEY}"
    
    response_news = requests.get(url_news)
    noticias = response_news.json()
    
    if isinstance(noticias, dict):
        if "error" in noticias:
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
        
        # Um micro-descanso de 1 segundo para não levar ban por "spam" na API gratuita
        time.sleep(1)

    df_noticias = pd.DataFrame(dados_processados)
    
    if df_noticias.empty:
        print("❌ Nenhuma notícia pôde ser processada.")
        return

    # Agrupa tirando a MÉDIA do sentimento do dia
    df_diario = df_noticias.groupby("Date")["Sentiment_Score"].mean().reset_index()
    
    caminho_arquivo = os.path.join('app', 'ml', 'pipeline', 'race_news.csv')
    df_diario.to_csv(caminho_arquivo, index=False)
    print(f"🚀 Sucesso! Arquivo '{caminho_arquivo}' gerado com sentimento matemático limpo.")

if __name__ == "__main__":
    csv_path = os.path.join('app', 'ml', 'pipeline', 'dados_transformados_V2.csv')
    dias = get_min_date_from_csv(csv_path)
    gerar_base_sentimento("RACE", dias_passado=dias)