FROM python:3.11-slim AS builder 
LABEL stage="builder"

WORKDIR /app 

COPY requirements.txt . 

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt 

# ----------- STAGE 2 --------------
FROM python:3.11-slim 

WORKDIR /app 

COPY --from=builder /install /usr/local 

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]