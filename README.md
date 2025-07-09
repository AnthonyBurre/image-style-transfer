Test app for artistic style transfer
============================


Usage
-----

It's easiest to run using Docker:

```shell
docker build -t style-transfer . 

docker run --rm -m "4g" -p 7860:7860 style-transfer
```