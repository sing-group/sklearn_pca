#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

conda activate sklearn_pca

if [ $? -gt 0 ]; then
    echo "Conda environment not found"
    exit 1
fi

python -m build

if [ $? -eq 0 ]; then
    echo "Build succeeded"
    VERSION=$(cat pyproject.toml | grep version | cut -f2 -d'=' | tr -d ' "')
    docker build ./ -t singgroup/sklearn_pca:${VERSION} --build-arg VERSION=${VERSION}
    if [ $? -eq 0 ]; then
        echo "Docker image built successfully"
        docker tag singgroup/sklearn_pca:${VERSION} singgroup/sklearn_pca:latest
    else
        echo "Docker image build failed"
        exit 1
    fi
else
    echo "Build failed"
    exit 1
fi
