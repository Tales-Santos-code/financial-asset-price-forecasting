import pytest
from fastapi.testclient import TestClient

# Importe o objeto 'app' do seu arquivo main
from app.api.main import app 

@pytest.fixture
def client():
    """
    Cria um cliente de teste do FastAPI que será injetado em todos os testes.
    """
    with TestClient(app) as c:
        yield c