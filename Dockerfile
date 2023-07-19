FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]



#CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]

# FROM tiangolo/uvicorn-gunicorn:python3.8-slim 

# WORKDIR /app 
# ENV DEBIAN_FRONTEND=noninteractive
# ENV MODULE_NAME=main
# ADD requirements.txt . 
# RUN pip install -r requirements.txt \    
#     && rm -rf /root/.cache 
# COPY . .
