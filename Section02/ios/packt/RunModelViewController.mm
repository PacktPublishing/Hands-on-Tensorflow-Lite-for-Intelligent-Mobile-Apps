// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#include "ios_image_load.h"

#define LOG(x) std::cerr
#define CHECK(x)                  \
  if (!(x)) {                     \
    LOG(ERROR) << #x << "failed"; \
    exit(1);                      \
  }

NSString* RunInferenceOnImage();

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
  NSString* inference_result = RunInferenceOnImage();
  self.urlContentTextView.text = inference_result;
}

@end

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
               << "' in bundle.";
  }
  return file_path;
}

NSString* RunInferenceOnImage() {
  NSString* graph = @"final_model";
  const int num_threads = 1;
  std::string input_layer_type = "float";
  std::vector<int> sizes = {2};

  const NSString* graph_path = FilePathForResourceName(graph, @"lite");

  std::unique_ptr<tflite::FlatBufferModel> model(
      tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]));
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << [graph UTF8String];
  }
    
  LOG(INFO) << "Loaded model " << [graph UTF8String];
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];
    
  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

    float* out = interpreter->typed_tensor<float>(input);
    out[0] = 1;
    out[1] = 1;
  
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke!";
  }

  float* output = interpreter->typed_output_tensor<float>(0);
  
    NSString* result = @(output[0]).description;
    
  return result;
}
