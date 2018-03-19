
def count_corrects(labels, predictions):
  num_correcs = 0
  for i, label in enumerate(labels):
    if label in predictions[i]:
      num_correcs += 1
  return num_correcs
