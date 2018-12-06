FROM python:3.5
# RUN apt-get update -y && \
# 	apt-get install -y python-pip python-dev

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./encoder /app/encoder
COPY ./data /app/data
COPY ./*.py /app/

EXPOSE 8000

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT [ "gunicorn", "--config", "gunicorn_config.py", "representativeStatementsFilter:app" ]
#ENTRYPOINT [ "waitress-serve",  "--port:8000", "representativeStatementsFilter:app" ]