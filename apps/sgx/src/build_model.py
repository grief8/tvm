#!/usr/bin/python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Creates a simple TVM modules."""

import os
from os import path as osp
import sys
import json
import shutil
import tvm
from tvm import relay
import tvm.relay.testing
import nets


def main():
    dshape = (3, 224, 224)
    # dshape = (192, 28, 28)
    batch_size = 1
    # net, params = relay.testing.resnet.get_workload(batch_size=batch_size, num_classes=10, num_layers=18,image_shape=dshape, dtype='float32')
    # net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, num_classes=10, image_shape=dshape, dtype='float32')
    net, params = nets.get_workload(batch_size=batch_size, num_classes=10, image_shape=dshape, dtype='float32')

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            net, target='llvm --system-lib',  params=params)

    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    lib.save(osp.join(build_dir, 'model.o'))
    with open(osp.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(osp.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))

    # with open(osp.join(os.getcwd(), 'config'), 'r') as f:
    #     config = json.load(f)["model_path"]
    # shutil.copyfile(osp.join(config, '../part2/model.o'), osp.join(build_dir, 'model1.o'))
    # shutil.copyfile(osp.join(config, '../part2/graph.json'), osp.join(build_dir, 'graph1.json'))
    # shutil.copyfile(osp.join(config, '../part2/params.bin'), osp.join(build_dir, 'params1.bin'))
    # shutil.copyfile(osp.join(config, '../part4/model.o'), osp.join(build_dir, 'model2.o'))
    # shutil.copyfile(osp.join(config, '../part4/graph.json'), osp.join(build_dir, 'graph2.json'))
    # shutil.copyfile(osp.join(config, '../part4/params.bin'), osp.join(build_dir, 'params2.bin'))
# def main():
#     build_dir = osp.abspath(sys.argv[1])
#     if not osp.isdir(build_dir):
#         os.makedirs(build_dir, exist_ok=True)
#     with open(osp.join(os.getcwd(), 'config'), 'r') as f:
#         config = json.load(f)["model_path"]
#     # shutil.copyfile(osp.join("/home/lifabing/tvm/apps/sgx/model", 'model.o'), osp.join(build_dir, 'model.o'))
#     # shutil.copyfile(osp.join("/home/lifabing/tvm/apps/sgx/model", 'graph.json'), osp.join(build_dir, 'graph.json'))
#     # shutil.copyfile(osp.join("/home/lifabing/tvm/apps/sgx/model", 'params.bin'), osp.join(build_dir, 'params.bin'))
#     shutil.copyfile(osp.join(config, 'model.o'), osp.join(build_dir, 'model.o'))
#     shutil.copyfile(osp.join(config, 'graph.json'), osp.join(build_dir, 'graph.json'))
#     shutil.copyfile(osp.join(config, 'params.bin'), osp.join(build_dir, 'params.bin'))
#     shutil.copyfile(osp.join(config, '../part2/model.o'), osp.join(build_dir, 'model1.o'))
#     shutil.copyfile(osp.join(config, '../part2/graph.json'), osp.join(build_dir, 'graph1.json'))
#     shutil.copyfile(osp.join(config, '../part2/params.bin'), osp.join(build_dir, 'params1.bin'))
#     shutil.copyfile(osp.join(config, '../part4/model.o'), osp.join(build_dir, 'model2.o'))
#     shutil.copyfile(osp.join(config, '../part4/graph.json'), osp.join(build_dir, 'graph2.json'))
#     shutil.copyfile(osp.join(config, '../part4/params.bin'), osp.join(build_dir, 'params2.bin'))


if __name__ == '__main__':
    main()
