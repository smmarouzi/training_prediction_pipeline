FROM mirror.gcr.io/library/python:3.10

RUN mkdir -p /app/model
WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
COPY server/*.py /app/
COPY server/entities /app/entities
COPY src /app/src

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$AIP_HTTP_PORT --log-level info --workers ${GUNICORN_WORKERS:-1} --timeout 90 app:app
