FROM python:3.11.7-slim-bullseye

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 8080
CMD ["flask","run","--host","0.0.0.0","--port","8080"]