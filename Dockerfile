# Usa uma imagem oficial, leve e padrão do Python (baseada em Debian)
FROM python:3.12-slim

# Define o diretório de trabalho padrão dentro do contêiner
WORKDIR /app

# Instala dependências do sistema e configura timezone
RUN apt-get update && \
    apt-get install -y libgomp1 tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copia os requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta da sua aplicação
COPY ./app ./app

# Expõe a porta que a API vai rodar
EXPOSE 8080

# Inicia o servidor Uvicorn diretamente (substitui o Lambda Handler)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]