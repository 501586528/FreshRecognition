FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cpu

# 安装系统依赖和设置时区
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get clean

# 设置国内 pip 源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install --upgrade pip

# 拷贝项目代码
COPY . /app
WORKDIR /app

# 安装你自己的依赖（torch 不要再写到 requirements.txt 里）
RUN pip install -r requirements.txt

EXPOSE 80
CMD ["python3", "app.py", "0.0.0.0", "80"]
