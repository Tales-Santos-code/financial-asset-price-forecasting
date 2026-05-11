import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.api.services.drift_detector import (
    load_production_logs,
    check_data_drift
)

@pytest.fixture
def mock_reference_data():
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=10),
        "Close": np.random.rand(10),
        "RSI_14": np.random.rand(10),
        "Target_Log_Return": np.random.rand(10)
    })

@pytest.fixture
def mock_current_data():
    return pd.DataFrame({
        "Date": pd.date_range("2026-02-01", periods=5),
        "Close": np.random.rand(5),
        "RSI_14": np.random.rand(5)
    })

PATCH_BASE = "app.api.services.drift_detector"

@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.get_s3_client")
def test_load_production_logs_sucesso(mock_get_s3, mock_read_json):
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'logs/1.json', 'LastModified': datetime.now()},
            {'Key': 'logs/2.json', 'LastModified': datetime.now()}
        ]
    }
    mock_get_s3.return_value = mock_s3
    mock_read_json.return_value = {"features_input": {"RSI_14": 0.5, "Close": 100}}
    
    df = load_production_logs("RACE")
    assert not df.empty
    assert "RSI_14" in df.columns

@patch(f"{PATCH_BASE}.write_html_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.load_production_logs")
@patch(f"{PATCH_BASE}.Report") # Patch no import local da função
def test_check_data_drift_estavel(mock_report_class, mock_load_logs, mock_read_csv, mock_write_html, mock_reference_data, mock_current_data):
    mock_read_csv.return_value = mock_reference_data
    mock_load_logs.return_value = mock_current_data
    
    # Mock do Report e do Snapshot retornado pelo .run()
    mock_report = MagicMock()
    mock_snapshot = MagicMock()
    mock_report.run.return_value = mock_snapshot
    
    # Mock das métricas
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.5},
                "value": {"share": 0.1}
            }
        ]
    }
    mock_snapshot.get_html_str.return_value = "<html></html>"
    mock_report_class.return_value = mock_report
    
    result = check_data_drift("RACE")
    assert result is False
    mock_write_html.assert_called_once()

@patch(f"{PATCH_BASE}.write_html_to_s3")
@patch(f"{PATCH_BASE}.disparar_retreino_github")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.load_production_logs")
@patch(f"{PATCH_BASE}.Report")
def test_check_data_drift_detectado(mock_report_class, mock_load_logs, mock_read_csv, mock_trigger, mock_write_html, mock_reference_data, mock_current_data):
    mock_read_csv.return_value = mock_reference_data
    mock_load_logs.return_value = mock_current_data
    
    mock_report = MagicMock()
    mock_snapshot = MagicMock()
    mock_report.run.return_value = mock_snapshot
    
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.5},
                "value": {"share": 0.8}
            }
        ]
    }
    mock_snapshot.get_html_str.return_value = "<html></html>"
    mock_report_class.return_value = mock_report
    
    result = check_data_drift("RACE")
    assert result is True
    mock_trigger.assert_called_once_with("RACE")
    mock_write_html.assert_called_once()