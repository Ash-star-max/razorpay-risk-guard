FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data and train
RUN python -m src.data_generator.generator
RUN python -m src.models.ensemble --train --evaluate

EXPOSE 8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
