# `sklearn_pca`

A Python script to perform PCA and generate bidimensional plots between the three first components.

A Docker image is available at [singgroup/sklearn_pca](https://hub.docker.com/r/singgroup/sklearn_pca).

## Motivation and example

Let's start with a TSV file like this (`test_data/data1.tsv`):

```tsv
sample  f1      f2      f3
s1      1        2       33
s2      4        5       6
s3      7        8       9
s4      10      11      12
```

And a CSV file with associated metadata (`test_data/metadata.csv`) fr the samples:
```tsv
sample,annotation1,annotation2
s1,A,C
s2,B,C
s3,A,D
s4,B,D
```

We would like to perform PCA on this table and colour samples by column `annotation1` (this is `--group`) and give them different shapes based on the `annotation2` column (this is `--shape`):

```shell
docker run --rm -v $(pwd):$(pwd) -w $(pwd) \
    singgroup/sklearn_pca pca \
        test_data/data1.tsv \
        --metadata test_data/metadata.csv \
        --group annotation1 \
        --shape annotation2 \
        --output_dir=test_data/test_docker_data1
```

In case your input data must be transposed, as with the `test_data/data2.tsv` file, use also `--transpose`:

```shell
docker run --rm -v $(pwd):$(pwd) -w $(pwd) \
    singgroup/sklearn_pca pca \
        test_data/data2.tsv \
        --transpose \
        --metadata test_data/metadata.csv \
        --group annotation1 \
        --shape annotation2 \
        --output_dir=test_data/test_docker_data2
```
