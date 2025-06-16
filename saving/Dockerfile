FROM python:3.12-slim

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/0.5.26/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
# ENV PATH="/app/.venv/bin:$PATH"
# ENV PATH="/root/.cargo/bin/:$PATH"
ENV PATH="/root/.local/bin/:$PATH"

RUN mkdir -p /app/app
WORKDIR /app/app

# ADD ./app/pyproject.toml ./
ADD ./pyproject.toml ./uv.lock ./

RUN uv sync --frozen

RUN echo "hello world"

ADD ./ ./

EXPOSE 8000

CMD [ "uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
