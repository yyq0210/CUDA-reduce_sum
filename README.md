v1: 原子加法规约
v2: warp内循环累加，warp间原子加法
v3: block内循环累加，block间原子加法
v4: block内树状累加，block间原子加法
v5: warp内树状累加（采用__shfl_down_sync()函数实现，warp间原子加法）
v6: warp内树状累加（采用__shfl_down_sync()函数实现，block内对warp_sum进行树状累加，block间原子加法
v7: warp内树状累加（采用__shfl_down_sync()函数实现，block内对warp_sum进行树状累加，block间继续采用规约代替原子加法