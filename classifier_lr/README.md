Example codes and experiments around Docker: Logistic Regression Classifier

- Dockerfile:



- requirements.txt:
list of needed packages to run app (e.g. python packages such as numpy, pandas,...)



How to run?
1) Install Docker (https://docs.docker.com/engine/install/): Docker Engine is available on a variety of Linux platforms, macOS and Windows.


2) Build: $docker build -f Dockerfile -t docker_classifier_lr .


3) Run: $docker run -it docker_classifier_lr /bin/bash


4) Go to the src directory (cd src/), activate the machine learning environment (source activate ml-env), and finally run python main file (python main.py)

 
