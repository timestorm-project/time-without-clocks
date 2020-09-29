# time-without-clocks

This source code release accompanies the paper with title "*[Activity in perceptual classification networks as a basis for human subjective time perception](https://www.nature.com/articles/s41467-018-08194-7)*", Nature Communications 10.1 (2019): 1-9.

### Instructions


#### 1. Install Caffe

For ubuntu/debian:

###### 1.1 Add this line in the end of the file:  ```/etc/apt/sources.list```.

```
deb [trusted=yes] http://ftp2.cn.debian.org/debian sid main contrib non-free
```

Please note that the ```non-free``` argument should be added if the CUDA version of caffe is needed.

###### 1.2 Then type:

```bash
sudo apt update
sudo apt install [ caffe-cpu | caffe-cuda ]
```

#### 2. Install other python dependencies

```bash
sudo pip3 install opencv-python sklearn
```

#### 3. Run demo system.
To run in real-time with a webcam as input, simply type

```bash
python3 demo_2018_nov.py
```

while to run with a set of given frames (given in *png* format), type

```bash
python3 demo_2018_nov.py -i <folder path>
```


## License

This project is licensed under the GLUv3 License - see the [LICENSE](LICENSE) file for details.
