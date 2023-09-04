/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

extern crate tvm_runtime;
extern crate ndarray;
extern crate rand;

use rand::Rng;
use std::{
    convert::TryFrom as _,
    io::{Read as _, Write as _},
    time::{SystemTime, UNIX_EPOCH, Duration},
    thread,
    env,
};
//  use image::{FilterType, GenericImageView};
use ndarray::{Array, Array4};

fn timestamp() -> i64 {
   let start = SystemTime::now();
   let since_the_epoch = start
       .duration_since(UNIX_EPOCH)
       .expect("Time went backwards");
   let ms = since_the_epoch.as_secs() as i64 * 1000i64 + (since_the_epoch.subsec_nanos() as f64 / 1_000_000.0) as i64;
   ms
}
#[cfg(target_env = "sgx")]
extern "C" {
    fn model1___tvm_module_startup();
    fn model2___tvm_module_startup();
}   
fn model1(){
    let shape = (1,3, 224, 224);
    // let shape = [(1, 32, 75, 75), (1, 128, 38, 38), (1, 512, 10, 10), (1, 512, 10, 10)];
    // let shape = [(1, 3, 224, 224), (1, 64, 38, 38), (1, 128, 19, 19), (1, 256, 10, 10)];
    let mut sy_time = SystemTime::now();
    let syslib = tvm_runtime::SystemLibModule::default();
    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_runtime::load_param_dict(params_bytes).unwrap();
    
    let graph = tvm_runtime::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_runtime::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);
    // println!("loading time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
    sy_time = SystemTime::now();
    let mut rng =rand::thread_rng();
    let mut ran = vec![];
    for _i in 0..shape.0*shape.1*shape.2*shape.3{
        ran.push(rng.gen::<f32>()*256.);
    }
    let x = Array::from_shape_vec(shape, ran).unwrap();
    // println!("generating time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
    exec.set_input("data", x.into());
    sy_time = SystemTime::now();
    exec.run();
    // let output = exec.get_output(0).unwrap();
    // println!("The shape: {:?}", output.shape());
    // println!("computing time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
    // println!("{:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
}

fn main() {

    model1();
    // model3();
   // println!("The index: {:?}", argmax);
   // println!("{:?}", sy_time.elapsed().unwrap().as_micros());
   // println!("{:#?}", output.data().as_slice());

}
