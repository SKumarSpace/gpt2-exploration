FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install datasets transformers==4.28.0
RUN pip install datasets
RUN pip install torch

WORKDIR /app
COPY app.py .

# Download the model
RUN python app.py --download

CMD ["python", "app.py"]