# 使用官方轻量 python 镜像
FROM --platform=linux/amd64 python:3.10-slim

# 安装基础工具和构建依赖
RUN apt-get update && \
    apt-get install -y gcc g++ libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# 设置国内 pip 镜像
RUN python -m pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 CPU 版 PyTorch + 其他依赖
RUN pip install \
    flask \
    flask_cors \
    pillow \
    numpy \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 拷贝代码
COPY . /app
WORKDIR /app

# 启动
EXPOSE 80
CMD ["python", "app.py"]