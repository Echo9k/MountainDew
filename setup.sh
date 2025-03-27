
mkdir -p model
wget https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int8-ov/resolve/main/openvino_language_model.xml      -O model/openvino_language_model.xml
wget https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int8-ov/resolve/main/openvino_language_model.bin      -O model/openvino_language_model.bin
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt