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
"""
eval.
"""
import os
import argparse
import random
import numpy as np
from dataset import create_dataset
from config import config
from mindspore import context
from mindspore.model_zoo.resnet import resnet50
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
from mindspore.communication.management import init
import moxing as mox  # 内置包  可以用来拷贝数据

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(enable_task_sink=True, device_id=device_id)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)

# 指定两个本地路径，因为data_url和train_url都是华为云上的路径
local_data_url = './cache/data'
local_ckpt_url = './cache/ckpt'

if __name__ == '__main__':
    context.set_context(enable_hccl=False)
    # 获得真正的checkpoint文件名
    if args_opt.checkpoint_path:
        checkpoint_file = os.path.join(local_ckpt_url,os.path.split(args_opt.checkpoint_path)[1])

    # 使用mox将数据从OBS中拷贝到容器中
    mox.file.copy_parallel(args_opt.data_url, local_data_url)
    mox.file.copy_parallel(args_opt.checkpoint_path, checkpoint_file)
    epoch_size = config.epoch_size
    net = resnet50(class_num=config.class_num)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)

    if args_opt.do_eval:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size)
        step_size = dataset.get_dataset_size()

        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(checkpoint_file)  # checkpoint路径名
            load_param_into_net(net, param_dict)
        net.set_train(False)

        model = Model(net, loss_fn=loss, metrics={'acc'})
        res = model.eval(dataset)
        print("result:", res, "ckpt=", local_ckpt_url)  # checkpoint路径名
