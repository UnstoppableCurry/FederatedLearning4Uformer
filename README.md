# FederatedLearning4Uformer
Implementing Federated Learning Code Based on UFormer
# 联邦 ZH
![b70007e68189330103ccfba2a578fe2](https://user-images.githubusercontent.com/65523997/230284540-c3130e7b-3d3a-4db5-84b1-6bba8fb6547c.png)
![2c2cb69123c9eead582f7d11e23fb44](https://user-images.githubusercontent.com/65523997/230285582-37389d03-6b10-4102-a811-b422170646e6.png)
![62689809b018872c6f72991a3c4d20c](https://user-images.githubusercontent.com/65523997/230285716-83e9e0d2-cef5-421c-989b-8d287e1cc615.png)
![f05ab04945f578be088f5f5d8929d41](https://user-images.githubusercontent.com/65523997/230285885-74f2fb85-0219-431a-8628-dfd39a25a2f1.png)
![8af13059ac5b1018878b1bbe70427fc](https://user-images.githubusercontent.com/65523997/230286163-c624ac2e-9493-46a5-af22-e742fa69b811.png)



目前项目支持tensorboard可视化全部评估结果,并支持在不同机器上面运行. 拟合性能经过验证到达通信次数1000次范围时模型收敛SOTA~

- 评估指标
    - RMSE 
        - 源代码-未更改  
        - tensorboard对应名称MSE
    - MAE 相对误差  
        - 代码实现时发现 模型没有拟合时 计算会出现NAN 和INF  数值溢出会直接报错无法计算. 为了计算 只能剔除 但是剔除掉的元素只能用0来替换 ,剔除会破坏原有shape 和 相对误差的计算
        - 这样求均值时会比实际值小 但是因为MAE不用做loss 所以不对模型拟合造成影响 但前期的MAE没有参考价值
    - PSNR 信噪比还是图像质量?
        - 源码中有提供计算函数 所以就加上了
- 结构介绍
    - server 端
        - 开启多个clinet 进行训练
        - 默认开启16个客户端 并选取其中的任意4个进行迭代
        - clinet 数量不能大于显卡数量 也就当前服务器只允许开启client的数量不能大于4
        - 并行训练是 用多进程实现的 , 区别于数据并行和模型并行的api 所以只能按照指定逻辑运行 ,单卡加载多个客户端 然后在多卡并行的逻辑异常难解决 并且单卡batchsize限制很大  建议想要解决这个问题 换一台更多卡的机器
        - client训练完毕后会在server进行合并求均值后进行评估,所有参数都将汇聚在cuda:0 设备上(不可更改设备)
    - clinet 客户端
        - 客户端 每个nepoch 都会在验证集上进行评估
        - 每次train 中的每次batch都会计算 指标,并在tensorboard中绘制  但没有均值
        - 每次评估时都会绘制 每次batch的 指标 与 每个指标在本次的均值 
    - data
        - 受限多进程实现并行训练,dataloader中的numworker参数只能设置为0, 大于0的任何数会导致 资源抢夺 死锁  程序不会报错也不会挂掉 但是会停止显卡的计算
        - 数据分组已经修复不均匀分配 但客户端过少时不建议使用(小于等于4)  参数 
        - 训练集 与验证集 8 2 分 不重复相互独立
    - 脚本运行
        - 目前只能使用编辑器来remote 运行, 服务端nohub命令 会导致报错 且报错无法解决 
        - 目前两个版本 server.py是多线程版 运行无问题 使用 server*多进程实现.py 会有概率死锁不建议使用*
    - 模型训练 与保存
        - 每组参数训练开始时必须设置不同的 模型保存路径 模型路径用以区别
        - 保存时 tensorboad 和模型保存路径都会在 savepath 路径下
        - 没有earlystoping ,人工对比最优通信次数 来选择最优模型即可
        - 每通信一次保存一个模型
    - tensorboard
        - client
            - 命名规则 前缀
                -  每个 客户端+ID+ 第N次通讯
                - 如果客户端0 在第1次通讯时被选取 其ID为 client0第0次通讯
                - 如果客户端0 在第1次通讯后直到第5次才被选取 那么其ID为 clinet0第5次通讯
                - 绘图需要下载cvs数据 人工校对次数
            - 训练 train 后缀命名规则
                - 横坐标X为step 每一个batchsize 步计数 直接就是 每一次
                - train*each*MAE
                    - 相对误差
                - train*each*MSE
                    - 均方根误差
                - train*each*PSNR
                    - 信噪比
                - train*epoch*
                    - 次数 无意义
            - 验证 test 后缀规则
                - test*MAE*Loss
                    - 同上
                - test*MSE*Loss
                    - 同上
                - test*PSNR*Loss
                    - 同上
                - test*epoch*eval*MAE*
                    - 横坐标x 每个epoch
                    - 纵坐标同上
                - test*epoch*eval*MSEc*
                    - 横坐标x 每个epoch
                    - 纵坐标同上
                - test*epoch*eval*psnr*
                    - 横坐标x 每个epoch
                    - 纵坐标同上
        - server
            - 命名规则 前缀
                - server
            - 验证 后缀规则
                - 每次
                    - 横坐标x 为  第多少次server端的验证 独立于通信次数计数(通信不重置)
                    - server*eval*each*MAE*LOSS
                    - server*eval*each*MSE*LOSS
                    - server*eval*each*PSNR*LOSS
                    - 纵坐标都一样同上
                - 每轮
                    - 横坐标x 为 第几次通信
                    - 纵坐标 为该次数下的均值
                    - server*epoch*mean*MAE*evl
                    - server*epoch*mean*MSE*evl
                    - server*epoch*mean*PSNR*evl

# 使用文档
![68b2b5d69fe6ce6e27a098cf362e69b](https://user-images.githubusercontent.com/65523997/230284811-4997114a-1d36-4de7-8f7b-21471f180415.png)
- 目前训练机器为4 x v100 32G  显卡越多加速越快
- train
    - nohup python server.py & 
    - 所有参数都在options.py 中 均有注释
    - 注意save*path 参数是保存模型文件与tensorboard文件*
- 查看训练日志
    - tail -f nohup.out
- test
    - python test.py  
    - 参数在文件中 需要配置测试集路径 和保存路径,运行前先清空对应保存路径中的文件夹内所有mat文件
    - 默认保存路径`/home/hyzb/user_wanghongzhou/SR_Test/results/ `
        - input
        - denoiseing
- 可视化命令
    - tensorboard --logdir= 你的路径
    - 你的路径对应 save*path即可*
    - 浏览器打开 ip:6006 即可访问
- 1000次 正确结果文件路径
    - `/home/hyzb/user_wanghongzhou/SR_Test/right_resume_False_num_comm_2000_client_16_gpuclient_4_batchsize_14_nepoch_1_IID_False_Neo等差_warmup_True`




#Federal EN

    -Evaluation indicators

    - RMSE 
        - Source Code - Unchanged
        - Tensorboard corresponding name MSE
        
    -MAE relative error
        -When implementing the code, it was found that when the model was not fitted, there would be overflow of NAN and INF values in the calculation, which would directly result in an error and prevent calculation In order to calculate, only elements that can be removed can be replaced with 0, which will disrupt the calculation of the original shape and relative error
        -This way, the mean will be smaller than the actual value, but because MAE does not need to do loss, it will not affect the model fitting, but the previous MAE has no reference value
    -PSNR signal-to-noise ratio or image quality?
        -The calculation function is provided in the source code, so it is added
    -Structure Introduction
    -Server side
        -Open multiple Clinets for training
        -By default, open 16 clients and select any 4 of them for iteration
        -The number of clientets cannot be greater than the number of graphics cards, which means that the current server only allows the number of clients to be enabled, which cannot be greater than 4
        -Parallel training is implemented using multiple processes, which is different from the API of data parallelism and model parallelism. Therefore, it can only run according to the specified logic. It is difficult to solve the logic exception of loading multiple clients on a single card and then parallelizing multiple cards. Additionally, the batch size of a single card is limited greatly. It is recommended to switch to a machine with more cards to solve this problem
        -After the client training is completed, it will be merged and averaged on the server for evaluation. All parameters will be aggregated on the cuda: 0 device (cannot be changed)
    -Clinet client
        -Each NEPOCH on the client will be evaluated on the validation set
        -Each batch in the train calculates metrics and plots them in the Tensorboard, but there is no mean value
        -During each evaluation, the indicators for each batch will be plotted along with the mean of each indicator in this evaluation
    - data
        -Restricted multi process implementation for parallel training, the numworker parameter in the dataloader can only be set to 0. Any number greater than 0 will cause resource grabbing deadlock, and the program will not report errors or hang up, but will stop the calculation of the graphics card
        -The parameter (less than or equal to 4) is not recommended when the data grouping has been fixed for uneven distribution but there are too few clients
        -The training set and validation set are independent of each other, with a score of 8.2 and no repetition
    -Script Run
        -Currently, only an editor can be used to remotely run. The server's nohub command can cause errors that cannot be resolved
        -Currently, the two versions of server.py are multithreaded versions that run without issues. Using server * for multi process implementation may result in a probability of deadlock. It is not recommended to use*
    -Model Training and Saving
        -At the beginning of each set of parameter training, different model save paths must be set to differentiate the model path
        -When saving, both the tensorboard and model save paths will be in the savepath path
        -Without early-stopping, manually comparing the optimal communication times to select the optimal model is sufficient
        -Save one model per communication
    - tensorboard
        - client
            -Naming Rule Prefix
                -Each client+ID+Nth communication
                -If client 0 is selected with its ID of client0 during the first communication, the 0 th communication
                -If client 0 is not selected until the 5th communication after the 1st communication, its ID is client0 for the 5th communication
                -Drawing requires downloading CVS data and manual proofreading times
            -Train suffix naming rules
                -The horizontal axis X represents the step, and each batch size step count directly represents each step
                - train*each*MAE
                    -Relative error
                - train*each*MSE
                    -Root mean square error
                - train*each*PSNR
                    -Signal-to-noise ratio
                - train*epoch*
                    -Times are meaningless
            -Verify test suffix rules
                - test*MAE*Loss
                    -Ditto
                - test*MSE*Loss
                    -Ditto
                - test*PSNR*Loss
                    -Ditto
                - test*epoch*eval*MAE*
                    -Horizontal coordinate x per epoch
                    -The vertical axis is the same as above
                - test*epoch*eval*MSEc*
                    -Horizontal coordinate x per epoch
                    -The vertical axis is the same as above
                - test*epoch*eval*psnr*
                    -Horizontal coordinate x per epoch
                    -The vertical axis is the same as above
            - server
                -Naming Rule Prefix
                    - server
                -Verify suffix rules
                    -Every time
                        -The x-axis represents the number of server side verifications independent of the communication count (communication is not reset)
                        - server*eval*each*MAE*LOSS
                       - server*eval*each*MSE*LOSS
                       - server*eval*each*PSNR*LOSS
                       -Same vertical axis as above
                       -Each round
                       -The horizontal axis x represents the number of communications
                       -The vertical axis is the mean value of this degree
                       - server*epoch*mean*MAE*evl
                       - server*epoch*mean*MSE*evl
                       - server*epoch*mean*PSNR*evl
# Using Documents
    - train
        - nohup python server.py & 
        -All parameters are annotated in options. py
        -Note that the save * path parameter is used to save the model file and tensorboard file*
    -View Training Log
        - tail -f nohup.out
    - test
        - python test.py  
        -The parameters need to be configured with the test set path and save path in the file. Before running, clear all mat files in the corresponding save path folder first
        -Default Save Path `/home/hyzb/user_ wanghongzhou/SR_ Test/results/ `
        - input
        - denoiseing
    -Visualization commands
        -Tensorboard -- logdir=Your path
        -Your path corresponds to save * path*
        -Open the browser to access ip: 6006
        -1000 correct result file paths
        - `/home/hyzb/user_ wanghongzhou/SR_ Test/right_ resume_ False_ num_ comm_ 2000_ client_ 16_ gpuclient_ 4_ batchsize_ 14_ nepoch_ 1_ IID_ False_ Neo equidistant_ warmup_ True`
