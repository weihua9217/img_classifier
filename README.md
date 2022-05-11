# Requirements
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboard==2.9.0

# Fruit Classifier
This is a Fruit Classifier, which backbone is ResNet


## Step1. Prepare your Dataset First
```
python arrange.py
```
## Step2. Train the ResNet:
```
python main.py --mode "train" --batch_size 64 --data_dir "./dataset/"
```

## Step3. To Test the result:
```
python main.py --mode "test" --data_dir "./dataset/"
```