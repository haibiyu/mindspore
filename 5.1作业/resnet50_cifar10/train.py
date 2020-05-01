# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train_imagenet."""
import os
import argparse
import random
import numpy as np
from dataset import create_dataset
from lr_generator import get_lr
from config import config
from mindspore import context
from mindspore import Tensor
from mindspore.model_zoo.resnet import resnet50
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset.engine as de
from mindspore.communication.management import init
import moxing as mox  # 内置包  可以用来拷贝数据

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
# parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
# parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
# parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(enable_task_sink=True, device_id=device_id)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)

# 指定两个本地路径，因为data_url和train_url都是华为云上的路径
local_data_url = './cache/data'
local_train_url = './cache/ckpt'

if __name__ == '__main__':
    if args_opt.do_eval:
        context.set_context(enable_hccl=False)
    else:
        if device_num > 1:
            context.set_context(enable_hccl=True)
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            auto_parallel_context().set_all_reduce_fusion_split_indices([140])
            init()
            # 为避免多卡上将数据下载到同一个位置，在这里需要做一个区分，根据device_id将数据下载到不同目录
            local_data_url = os.path.join(local_data_url,str(device_id))
        else:
            context.set_context(enable_hccl=False)
    # 使用mox将数据从OBS中拷贝到容器中
    mox.file.copy_parallel(args_opt.data_url,local_data_url)
    mox.file.copy_parallel(args_opt.train_url,local_train_url)
    epoch_size = config.epoch_size
    net = resnet50(class_num=config.class_num)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)


    if args_opt.do_train:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True,
                                 repeat_num=epoch_size, batch_size=config.batch_size)
        step_size = dataset.get_dataset_size()

        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                           lr_decay_mode='poly'))
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                       config.weight_decay, config.loss_scale)

        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})

        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if config.save_checkpoint and (device_num == 1 or device_id == 0):  # 单机 或 多卡只用第0卡去执行checkpoint
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="resnet", directory=local_train_url, config=config_ck)
            cb += [ckpt_cb]
        model.train(epoch_size, dataset, callbacks=cb)
        # 输出结果要从容器中拷贝到OBS上
        if config.save_checkpoint and (device_num == 1 or device_id == 0):
            mox.file.copy_parallel(local_train_url,args_opt.train_url)