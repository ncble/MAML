# MAML-TensorFlow
An elegant and efficient implementation for ICML2017 paper: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)


# Highlights
- the code are adopted from the [github repository](https://github.com/dragen1860/MAML-TensorFlow). Our modifications are: the change of **backbone networks**, the use of **data augmentation**, the other changes for our ablation experiments, etc.

- further improvements are coming...


# How-TO
1. Download mini-Imagenet from [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) and extract them as :
```shell
	miniimagenet/	
	├── images	
		├── n0210891500001298.jpg  		
		├── n0287152500001298.jpg 		
		...		
	├── test.csv	
	├── val.csv	
	└── train.csv	
	└── proc_images.py
	
```

then replace the `path` by your actual path in `data_generator.py`:
```python
		metatrain_folder = config.get('metatrain_folder', '/hdd1/liangqu/datasets/miniimagenet/train')
		if True:
			metaval_folder = config.get('metaval_folder', '/hdd1/liangqu/datasets/miniimagenet/test')
		else:
			metaval_folder = config.get('metaval_folder', '/hdd1/liangqu/datasets/miniimagenet/val')
```	

2. resize all raw images to 84x84 size by
```shell
python proc_images.py
```

3. train
```shell
python main.py

```

4. test
```shell
python main.py -t
```


