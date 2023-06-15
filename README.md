# stylegan2风格迁移快速上手
## 环境搭建
- git clone源码

    ```
    git clone https://github.com/hDluffy/stylegan2.git
    ```
    > 同时下载源github中提供的预训练模型stylegan2-ffhq-config-f.pkl，放置networks目录下

- 安装训练环境</br>

    conda配置
    ```
    conda create --name stylegan2-tf python=3.6
    
    source activate stylegan2-tf
    
    ##注意cuda版本与系统安装版本无关
    conda install tensorflow-gpu=1.14
    
    pip install scipy==1.3.3
    pip install requests==2.22.0
    pip install Pillow==6.2.1
    pip install typer
    ```
    docker环境
    
    ```
    docker build .
    ```


- 数据处理

    1024x1024对齐后的raw数据,通过dataset_tool.py转为tfrecords数据集
    ```
    #转为tfrecords数据集
    ./make_dataset.sh
    
    #转为tfrecords数据集的具体指令
    python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images
    #可以查看数据
    python dataset_tool.py display ~/datasets/my-custom-dataset
    ```

- 模型训练</br>

    ```
    ./train.sh
    
    #CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py --num-gpus=4 --data-dir=~/datasets --config=config-f --dataset=my-custom-dataset --mirror-augment=true
    ```
    注:</br>
    可能的错误undefined symbol: _ZN10tensorflow12OpDefBuilder6OutputESs</br>
    解决方案：https://blog.csdn.net/zaf0516/article/details/103618601</br>
    可能的错误#error "C++ versions less than C++11 are not supported</br>解决方案：https://blog.csdn.net/qq1483661204/article/details/105442426
    > 1.通过设置./traing/training_loop.py中的resume_pkl路径，可以修改预训练模型</br>
    > 2.建议注释掉评估，可以加快训练【metrics】

- 测试及批量处理

    ```
    #task='GenShow',显示不同层融合的效果对比；task='GenBatch',进行特定层融合效果的批量生成；
    ./gen_images.sh
    
    #处理完的数据，可以通过指令copy到本地处理
    scp username@ip:/path/data /mnt/g/DataSets