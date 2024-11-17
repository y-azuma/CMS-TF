FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

ENV DEBIAN_FRONTED=noninteractive
ENV PATH=$PATH:/usr/local/scala/scala-2.11.7/bin
RUN mkdir /.cache
RUN chown 1000:1000 /.cache
RUN chown 1000:1000 /etc
ENV python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install \
    nvidia-cudnn-cu11==8.6.0.163 \
    tensorflow-datasets==4.9.2 \
    tensorflow-image-models==0.0.11 \
    tfimm==0.2.14\
    timm==1.0.11\
    dataclasses-json==0.6.7 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.5 \
    typing_extensions==4.5.0\
    typeguard==2.7.0 
    
RUN mkdir -p /ws/cms

WORKDIR /ws/cms