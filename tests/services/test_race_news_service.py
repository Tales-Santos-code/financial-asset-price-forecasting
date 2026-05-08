import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ajuste o caminho do import caso o nome do arquivo seja diferente
from app.api.services.Race_news_service import (
    get_min_date_from_s3,
    analisar_sentimento_finbert,
    gerar_base_sentimento
)

# Caminho base para os Mocks
PATCH_BASE = "app.api.services.Race_news_service"


# ==========================================
# 1. TESTES DA LEITURA DE DATA MÍNIMA
# ==========================================

@patch(f"{PATCH_BASE}.read_csv_from_s3")
def test_get_min_date_from_s3_sucesso(mock_read_s3):
    # Finge que o arquivo Master tem dados de exatos 10 dias atrás
    hoje = pd.Timestamp.now().normalize()
    data_antiga = hoje - pd.Timedelta(days=10)
    
    mock_df = pd.DataFrame({"Date": [data_antiga.strftime("%Y-%m-%d")]})
    mock_read_s3.return_value = mock_df
    
    resultado = get_min_date_from_s3("RACE")
    
    # Deve retornar exatamente 10 dias
    assert resultado == 10

@patch(f"{PATCH_BASE}.read_csv_from_s3")
def test_get_min_date_from_s3_vazio(mock_read_s3):
    # Finge que o S3 não tem o arquivo master
    mock_read_s3.return_value = None
    
    resultado = get_min_date_from_s3("RACE")
    
    # Deve acionar o fallback de segurança (Carga total de 360 dias)
    assert resultado == 360


# ==========================================
# 2. TESTES DA INTELIGÊNCIA ARTIFICIAL (FinBERT)
# ==========================================

@patch(f"{PATCH_BASE}.hf_client")
def test_analisar_sentimento_finbert_positivo(mock_hf_client):
    # Simula a resposta da Hugging Face para uma notícia boa
    mock_hf_client.text_classification.return_value = [
        {"label": "positive", "score": 0.85},
        {"label": "neutral", "score": 0.10},
        {"label": "negative", "score": 0.05}
    ]
    
    score = analisar_sentimento_finbert("Ferrari bate recorde de vendas")
    
    # A sua lógica subtrai o valor negativo do positivo (0.85 - 0.05 = 0.80)
    # Usamos pytest.approx() para lidar com a imprecisão de casas decimais do Python (0.799999...)
    assert score == pytest.approx(0.80)

@patch(f"{PATCH_BASE}.hf_client")
def test_analisar_sentimento_finbert_negativo(mock_hf_client):
    # Simula a resposta para uma notícia ruim
    mock_hf_client.text_classification.return_value = [
        {"label": "negative", "score": 0.90}
    ]
    
    score = analisar_sentimento_finbert("Ações despencam após escândalo")
    
    # Como é negativo, o script inverte o sinal matematicamente
    assert score == pytest.approx(-0.90)

@patch(f"{PATCH_BASE}.time.sleep") # Mockamos o sleep para o teste não demorar 3 segundos!
@patch(f"{PATCH_BASE}.hf_client")
def test_analisar_sentimento_finbert_retry_503(mock_hf_client, mock_sleep):
    # Finge que a HF dá Erro 503 na primeira tentativa, mas funciona na segunda!
    mock_hf_client.text_classification.side_effect = [
        Exception("503 Server Error: Model is loading"),
        [{"label": "positive", "score": 0.5}]
    ]
    
    score = analisar_sentimento_finbert("Notícia qualquer")
    
    # Verificações
    assert score == 0.5
    # O time.sleep(3) DEVE ter sido chamado para dar tempo do servidor respirar
    mock_sleep.assert_called_once_with(3)
    # A API da HF foi chamada duas vezes
    assert mock_hf_client.text_classification.call_count == 2

@patch(f"{PATCH_BASE}.hf_client")
def test_analisar_sentimento_finbert_erro_fatal(mock_hf_client):
    # Finge um erro que não é de servidor (ex: Token inválido)
    mock_hf_client.text_classification.side_effect = Exception("401 Unauthorized")
    
    score = analisar_sentimento_finbert("Notícia qualquer")
    
    # Deve estourar o limite, cair no except final e retornar neutro (0) sem tentar denovo
    assert score == 0


# ==========================================
# 3. TESTES DA ENGENHARIA DE DADOS (Finnhub + S3)
# ==========================================

@patch(f"{PATCH_BASE}.requests.get")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
def test_gerar_base_sentimento_limite_api(mock_write_s3, mock_requests):
    # Finge que o Finnhub estourou a cota
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "API limit reached"}
    mock_requests.return_value = mock_response
    
    gerar_base_sentimento("RACE", 5)
    
    # Garante que o pipeline abortou e NÃO tentou salvar lixo no S3
    mock_write_s3.assert_not_called()

@patch(f"{PATCH_BASE}.time.sleep") # Pula o delay de 1 seg por notícia
@patch(f"{PATCH_BASE}.analisar_sentimento_finbert")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.requests.get")
def test_gerar_base_sentimento_caminho_feliz(mock_requests, mock_write_s3, mock_read_s3, mock_finbert, mock_sleep):
    # 1. Simula Finnhub devolvendo duas notícias no mesmo dia
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"headline": "Notícia 1", "datetime": 1715000000}, # Exemplo de timestamp
        {"headline": "Notícia 2", "datetime": 1715000000}
    ]
    mock_requests.return_value = mock_response
    
    # 2. Finge que a IA deu +1.0 pra primeira e -0.5 pra segunda
    mock_finbert.side_effect = [1.0, -0.5]
    
    # 3. Finge que o S3 já tinha um histórico antigo para vermos a mesclagem
    df_antigo = pd.DataFrame([{"Date": "2024-01-01", "Sentiment_Score": 0.1}])
    mock_read_s3.return_value = df_antigo
    
    # AÇÃO
    gerar_base_sentimento("RACE", 5)
    
    # VERIFICAÇÕES
    mock_write_s3.assert_called_once()
    
    # Extrai o DataFrame que seria salvo no S3
    df_salvo = mock_write_s3.call_args[0][2]
    
    # Garante que a mesclagem funcionou (O dado de 2024 + O dado novo)
    assert len(df_salvo) == 2
    
    # Garante que ele calculou a MÉDIA do dia novo corretamente
    # (1.0 - 0.5) / 2 = 0.25
    nova_linha = df_salvo.iloc[-1]
    assert nova_linha["Sentiment_Score"] == 0.25