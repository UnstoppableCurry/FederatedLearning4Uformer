# FederatedLearning4Uformer
Implementing Federated Learning Code Based on UFormer
# 联邦
![b70007e68189330103ccfba2a578fe2](https://user-images.githubusercontent.com/65523997/230284540-c3130e7b-3d3a-4db5-84b1-6bba8fb6547c.png)
![2c2cb69123c9eead582f7d11e23fb44](https://user-images.githubusercontent.com/65523997/230285582-37389d03-6b10-4102-a811-b422170646e6.png)

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

