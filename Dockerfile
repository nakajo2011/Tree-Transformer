
FROM continuumio/miniconda3:latest

RUN conda install pytorch torchvision python-graphviz jupyter jupyterlab scikit-learn pandas -c pytorch -c conda-forge
RUN pip3 install transformers svgling cairosvg nltk
RUN mkdir -p /conda
WORKDIR /conda

ENTRYPOINT ["jupyter", "lab", "--port", "8888", "--ip=0.0.0.0", "--allow-root"]
# docker build --rm=true -t condatorch .