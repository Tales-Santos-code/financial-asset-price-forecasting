# 🚨 O BOTÃO DO HOMEM MORTO: Agenda o desligamento da máquina para daqui a 60 min.
            # Se a máquina travar por falta de RAM, ela vai se auto-desligar!
            set -e

            sudo shutdown -P +240 "Desligamento de segurança ativado pelo GitHub Actions"
            
            cd /home/ubuntu
            
            # 📁 Gerencia o Repositório
            if [ ! -d "financial-asset-price-forecasting" ]; then
              echo "Clonando o repositório..."
              git clone https://github.com/Tales-Santos-code/financial-asset-price-forecasting.git
            fi
            
            cd financial-asset-price-forecasting
            echo "Limpando arquivos temporários e forçando atualização..."
            git checkout .
            git clean -fd -e venv/
            git pull origin main
            
            # garante que o docker vai subir com a configuração mais recente
            docker compose -f app/ml/server/docker-compose-ec2.yaml down

            # 🐳 Garante que o MLflow (Cartório) está online antes do treino
            echo "Subindo o MLflow via Docker Compose..."
            docker compose -f app/ml/server/docker-compose-ec2.yaml up -d

            # Dá um tempo para o banco e o servidor MLflow iniciarem completamente
            echo "Aguardando o MLflow ficar online..."
            for i in {1..20}; do
              if curl -s -f http://127.0.0.1:5000/ping > /dev/null; then
                echo "✅ MLflow está online!"
                break
              fi
              echo "⏳ Aguardando MLflow... ($i/20)"
              sleep 10
            done
            
            # 🐍 Configura e ativa o ambiente virtual do Python
            echo "Configurando dependências do Python..."
            if [ ! -d "venv" ]; then
                python3 -m venv venv
            fi
            source venv/bin/activate
            pip install --upgrade pip
            pip install --prefer-binary -r app/ml/server/requirements-ml.txt
            
            symbol=NVDA
            # 🚀 O GRANDE MOMENTO: Roda a Fábrica de Experimentos
            echo "Iniciando o retreino para: $symbol"
            PYTHONPATH=. PYTHONUNBUFFERED=1 timeout 210m python app/ml/training/random_search.py --symbol $symbol --verbose
            
            # 🛑 Se o código chegou até aqui, foi um sucesso! Cancela o desligamento programado.
            sudo shutdown -c
            echo "Treino concluído com sucesso. Desligamento emergencial cancelado."


#["RACE", "NVDA", "AAPL", "VALE3.SA", "ITSA4.SA", "WEGE3.SA", "^GSPC"]