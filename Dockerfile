FROM python:3.8.10
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./app /code/app
WORKDIR /code
EXPOSE $PORT
CMD ["uvicorn", "app.appfile:app", "--host=0.0.0.0", "--reload", "--port=$PORT"]
