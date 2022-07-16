# robust-scanner-tf2
tensorflow2 implementation of robust scanner

```python
Version: 0.0.1
```

**LOCAL ENVIRONMENT**  

```python
OS          : Ubuntu 20.04.3 LTS       
Memory      : 23.4 GiB 
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.36.8
```

# Environment Setup

**python requirements**

* **pip requirements**: ```pip install -r requirements.txt``` 

> Its better to use a virtual environment 
> OR use conda-

* **conda**: use environment.yml: ```conda env create -f environment.yml```

# Paper documentation


# Data Creation
* git clone this repo
* change directory to scripts folder: ```cd scripts```
* execute data.py: 

```
usage: Robust Scanner tfrecord Dataset Creation Script [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--label_max_len LABEL_MAX_LEN] [--num_process NUM_PROCESS] data_txt vocab_txt save_path

positional arguments:
  data_txt              Path of the data.txt file holding absolute image path and word by tab separation
  vocab_txt             Path of the vocab.txt file holding the unicodes to use
  save_path             Path of the directory to save the tfrecord dataset

optional arguments:
  -h, --help            show this help message and exit
  --img_height IMG_HEIGHT
                        the desired height of the image:default=32
  --img_width IMG_WIDTH
                        the desired width of the image:default=256
  --label_max_len LABEL_MAX_LEN
                        maximum length for the text label:default=40
  --num_process NUM_PROCESS
                        number of processes to be used:default=16
```


# TODO
- [ ] add paper documentation in readme file. (The original paper)
- [ ] add training script documentation
- [ ] prepare configureable training script
- [ ] test arabic data storing
- [ ] add documented notebooks in datasets