FROM public.ecr.aws/lambda/python:3.12

# 1. Instala a biblioteca de sistema operacional exigida pelo XGBoost
RUN dnf install -y libgomp

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ${LAMBDA_TASK_ROOT}/app

CMD ["app.api.main.handler"]