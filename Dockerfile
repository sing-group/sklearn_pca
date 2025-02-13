FROM python:3.12.6-slim-bullseye

ARG VERSION

COPY dist/sklearn_pca-${VERSION}.tar.gz /tmp/sklearn_pca-${VERSION}.tar.gz
        
RUN pip install --upgrade pip && pip install /tmp/sklearn_pca-${VERSION}.tar.gz