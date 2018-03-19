import json
import os

from bunch import Bunch


def get_config_from_json(json_file):
  """
  Get the config from a json file
  :param json_file:
  :return: config(namespace) or config(dictionary)
  """
  # parse the configurations from the config json file provided
  with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

  # convert the dictionary to a namespace using bunch lib
  config = Bunch(config_dict)

  return config, config_dict


def count_file_lines(file_path):
  """
  Counts the number of lines in a file using wc utility.
  :param file_path: path to file
  :return: int, no of lines
  """
  count = 0
  with open(file_path) as f:
    for line in f:
      count += 1
  return count


def process_config(json_file):
  config, _ = get_config_from_json(json_file)
  config.num_loc = count_file_lines(os.path.join('data/', config.data_set, "locations.txt"))
  config.num_user = count_file_lines(os.path.join('data/', config.data_set, "users.txt"))
  config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/", config.cell + '_' + config.data_set)
  config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/", config.cell + '_' + config.data_set)
  try:
    os.rmdir(os.path.join("../experiments", config.exp_name))
  except:
    pass
  return config
