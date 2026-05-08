import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ajuste este import para o caminho real da sua classe
from app.api.services.finance_service import FinanceService

# ==========================================
# FIXTURES E DADOS FALSOS (MOCKS)
# ==========================================
@pytest.fixture
def mock_yf_history_df():
    """
    Simula o DataFrame retornado pelo yfinance.
    Nota crucial: O yfinance sempre retorna o índice (Date) com fuso horário (tz-aware).
    Nós simulamos isso aqui para testar se o nosso código limpa esse fuso horário.
    """
    df = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [101.0, 103.0],
        "Low": [99.0, 101.0],
        "Close": [100.5, 102.5],
        "Volume": [1000, 1500],
        "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0]
    })
    # Criando o índice de datas com fuso horário (UTC) igual ao Yahoo Finance
    df.index = pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC")
    return df

@pytest.fixture
def mock_yf_macro_df():
    """Simula o retorno da busca macroeconômica."""
    df = pd.DataFrame({
        "^GSPC": [4500.0, 4510.0],
        "^VIX": [15.0, 14.5],
        "EURUSD=X": [1.1, 1.11]
    })
    df.index = pd.DatetimeIndex(["2026-01-01", "2026-01-02"])
    return df


# Caminho base para injetar os Mocks
PATCH_BASE = "app.api.services.finance_service"

# ==========================================
# 1. TESTE DE INICIALIZAÇÃO
# ==========================================
@patch(f"{PATCH_BASE}.yf.Ticker")
def test_finance_service_init(mock_ticker):
    # Ação
    service = FinanceService(ticker="AAPL")
    
    # Verificação
    assert service.ticker_symbol == "AAPL"
    assert "^GSPC" in service.ticker_macro
    mock_ticker.assert_called_once_with("AAPL")


# ==========================================
# 2. TESTES DE DADOS HISTÓRICOS (get_historical_data)
# ==========================================

# Teste 2.1: Carga Total (Sem Ponteiro no S3)
@patch(f"{PATCH_BASE}.yf.Ticker")
@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.write_json_to_s3")
def test_get_historical_data_carga_total(mock_write_s3, mock_read_s3, mock_ticker_class, mock_yf_history_df):
    # Setup Mocks
    mock_read_s3.return_value = None  # S3 vazio (não tem ponteiro)
    
    mock_instancia_ticker = MagicMock()
    mock_instancia_ticker.history.return_value = mock_yf_history_df
    mock_ticker_class.return_value = mock_instancia_ticker
    
    service = FinanceService("RACE")
    
    # Ação: Carga total e atualizando ponteiro
    df_resultado = service.get_historical_data(full=True, use_checkpoint=True)
    
    # Verificações
    assert not df_resultado.empty
    
    # 1. O yfinance deveria ter sido chamado pedindo TODO o histórico (period="max")
    mock_instancia_ticker.history.assert_called_once_with(
        period="max", interval="1d", prepost=True, actions=False
    )
    
    # 2. O índice deve estar sem o fuso horário (tz-naive)
    assert df_resultado.index.tz is None
    
    # 3. O S3 deve ter sido atualizado com a data mais recente do nosso mock ("2026-01-02")
    mock_write_s3.assert_called_once()
    args, kwargs = mock_write_s3.call_args
    
    # CORREÇÃO AQUI 👇
    # Os argumentos posicionais (args) são: (bucket, pointer_key, payload_dicionario)
    # Então o dicionário com a data está na posição args[2]
    assert args[2]["last_date"] == "2026-01-02"


# Teste 2.2: Carga Incremental (Ponteiro existe no S3)
@patch(f"{PATCH_BASE}.yf.Ticker")
@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.write_json_to_s3")
def test_get_historical_data_carga_incremental(mock_write_s3, mock_read_s3, mock_ticker_class, mock_yf_history_df):
    # Setup Mocks
    # Fingimos que o último download parou no dia 01/01/2026
    mock_read_s3.return_value = {"last_date": "2026-01-01"} 
    
    mock_instancia_ticker = MagicMock()
    mock_instancia_ticker.history.return_value = mock_yf_history_df
    mock_ticker_class.return_value = mock_instancia_ticker
    
    service = FinanceService("RACE")
    
    # Ação
    service.get_historical_data(full=True, use_checkpoint=True)
    
    # Verificações
    # A sacada principal: A data do ponteiro era 01/01. O yfinance tem que ser chamado a partir do dia 02/01!
    mock_instancia_ticker.history.assert_called_once_with(
        start="2026-01-02", interval="1d", prepost=True, actions=False
    )


# Teste 2.3: Filtro de Colunas (full=False)
@patch(f"{PATCH_BASE}.yf.Ticker")
@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.write_json_to_s3")
def test_get_historical_data_filtro_colunas(mock_write_s3, mock_read_s3, mock_ticker_class, mock_yf_history_df):
    # Setup Mocks
    mock_read_s3.return_value = None
    mock_instancia_ticker = MagicMock()
    mock_instancia_ticker.history.return_value = mock_yf_history_df
    mock_ticker_class.return_value = mock_instancia_ticker
    
    service = FinanceService("RACE")
    
    # Ação: Chamando com full=False
    df_resultado = service.get_historical_data(full=False, use_checkpoint=False)
    
    # Verificações
    # O mock continha colunas como "Dividends" e "Stock Splits", elas DEVEM ter sido apagadas
    assert "Dividends" not in df_resultado.columns
    assert list(df_resultado.columns) == ["Open", "High", "Low", "Close", "Volume"]
    
    # O S3 NÃO deve ter sido chamado (pois use_checkpoint=False)
    mock_write_s3.assert_not_called()


# Teste 2.4: Proteção contra Retorno Vazio (Feriado ou erro na bolsa)
@patch(f"{PATCH_BASE}.yf.Ticker")
@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.write_json_to_s3")
def test_get_historical_data_vazio(mock_write_s3, mock_read_s3, mock_ticker_class):
    # Setup Mocks
    mock_read_s3.return_value = {"last_date": "2026-01-01"}
    
    mock_instancia_ticker = MagicMock()
    # Finge que hoje é sábado e a bolsa não retornou nada
    mock_instancia_ticker.history.return_value = pd.DataFrame() 
    mock_ticker_class.return_value = mock_instancia_ticker
    
    service = FinanceService("RACE")
    
    # Ação
    df_resultado = service.get_historical_data(full=True, use_checkpoint=True)
    
    # Verificações
    assert df_resultado.empty
    # Se veio vazio, NÃO podemos atualizar o ponteiro no S3 pra não estragar a lógica!
    mock_write_s3.assert_not_called()


# ==========================================
# 3. TESTES MACROECONÔMICOS (get_macro_data)
# ==========================================
@patch(f"{PATCH_BASE}.yf.download")
@patch(f"{PATCH_BASE}.yf.Ticker")
def test_get_macro_data(mock_ticker_class, mock_yf_download, mock_yf_macro_df):
    # Setup Mocks
    mock_yf_download.return_value = mock_yf_macro_df
    
    service = FinanceService("RACE")
    
    # Ação
    df_resultado = service.get_macro_data(min_date="2026-01-01", max_date="2026-01-10")
    
    # Verificações
    assert not df_resultado.empty
    
    # Garante que ele passa as datas corretamente para o yfinance
    # Lembre-se: O código soma 5 dias na data máxima pra garantir cobertura!
    mock_yf_download.assert_called_once()
    args, kwargs = mock_yf_download.call_args
    
    assert kwargs["start"] == "2026-01-01"
    # 2026-01-10 + 5 dias = 2026-01-15
    assert kwargs["end"] == pd.to_datetime("2026-01-15")