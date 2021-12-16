"""
tf_helper.py
purpose: helper functions to save a model when using multiworkermirroredstrategy
taken from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading
note that the model will always be saved to write_model_path/saved_model.pb
read more about model types here: https://stackoverflow.com/questions/59887312/when-to-use-the-ckpt-vs-hdf5-vs-pb-file-extensions-in-tensorflow-model-savin
https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
"""

import os
import tensorflow as tf

def _is_chief(task_type, task_id):
  # Note: there are two possible `TF_CONFIG` configuration.
  #   1) In addition to `worker` tasks, a `chief` task type is use;
  #      in this case, this function should be modified to
  #      `return task_type == 'chief'`.
  #   2) Only `worker` task type is used; in this case, worker 0 is
  #      regarded as the chief. The implementation demonstrated here
  #      is for this case.
  # For the purpose of this colab section, we also add `task_type is None`
  # case because it is effectively run with only single worker.
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

"""
example use:

model_path = '/path/to/model/'

task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
write_model_path = write_filepath(model_path, task_type, task_id)
print('Total Time:', time()-start)
model.save(write_model_path)

#remove the temporary/non-chief worker models:
if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(os.path.dirname(write_model_path))
"""

if __name__ == "__main__":
    main()
