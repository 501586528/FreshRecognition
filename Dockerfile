# 使用官方 Python Slim 镜像，适用于 PyTorch
FROM python:3.10-slim

# 设置时区为上海（可选）
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get clean

# 安装 HTTPS、基础构建依赖
RUN apt-get update && \
    apt-get install -y ca-certificates gcc g++ make wget curl && \
    apt-get clean

# 设置国内 pip 源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip

# 拷贝项目文件
COPY . /app
WORKDIR /app

# 安装依赖
RUN pip install -r requirements.txt

# 容器开放端口（应与微信云托管配置一致）
EXPOSE 80

# 启动命令（确保 app.py 是入口文件）
CMD ["python3", "app.py", "0.0.0.0", "80"]