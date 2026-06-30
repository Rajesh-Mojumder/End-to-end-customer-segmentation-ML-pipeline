FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Train the model during Docker build so model.joblib exists at runtime
RUN python main.py

EXPOSE 8080

CMD ["python3", "app.py"]
