FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . .
# pip 同样走阿里云 PyPI 镜像，避免境外源下载 langchain 等大包时卡死
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install ".[qdrant,pg]" -i https://mirrors.aliyun.com/pypi/simple/

EXPOSE 8000
# 健康探针不依赖 curl（镜像无需 apt）：python:3.11-slim 自带 python，直接用它探 /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/healthz',timeout=4).getcode()==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
