FROM python:3.6
ADD ./untitled /coda
WORKDIR /coda
RUN mkdir -p /home/itc/ai_resource
RUN pip install -r /coda/requirements.txt -i https://pypi.douban.com/simple
CMD ["python","/coda/cws_AI/check/start.py"]
