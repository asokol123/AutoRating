FROM python:3.8

RUN mkdir /app
COPY src/requirements.txt /app

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

USER nobody

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
