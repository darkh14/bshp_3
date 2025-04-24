FROM python:3.12.4
 
LABEL authors="nickgott"

WORKDIR /

EXPOSE 8061:8061

COPY . .

RUN python3 -m pip install --root-user-action=ignore --upgrade pip && python3 -m  pip install --root-user-action=ignore -r requirements.txt

ENTRYPOINT ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8061"]