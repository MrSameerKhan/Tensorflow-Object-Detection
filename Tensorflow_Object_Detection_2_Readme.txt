# GPU or CPU
import tensorflow as tf
tf.test.is_gpu_available()

# Git Clone
https://github.com/tensorflow/models

# Protobuf Installation
1. from models/research
mkdir protoc
cd protoc
2. download from here
https://github.com/protocolbuffers/protobuf/releases
unzip 
3. run the command from models/research
protoc_3.14/protoc-3.14.0-linux-x86_64/bin/protoc object_detection/protos/*.proto --python_out=.
4. Add libraries to PYTHONPATH
gedit ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/sameer/Videos/Tensorflow_2/models
# export PYTHONPATH=:/home/sameer/Videos/Tensorflow_2/models/research:/home/sameer/Videos/Tensorflow_2/models/research/slim
source ~/.bashrc

# Test Installation
python3 object_detection/builders/model_builder_tf2_test.py


