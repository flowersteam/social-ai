# Dockerfile for the Huggingface spaces Demo
FROM python:3.7


WORKDIR /code

# Install graphviz
RUN apt-get update && \
    apt-get install -y graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN chmod -R 777 /code

RUN pip install --upgrade -r web_demo/requirements.txt
RUN pip install -e gym-minigrid

#EXPOSE 7860

CMD ["python", "web_demo/app.py"]


# docker build -t sai_demo  -f web_demo/Dockerfile .
# docker run -p 7860:7860 sai_demo
