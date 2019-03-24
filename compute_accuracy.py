import sys
import numpy as np

if len(sys.argv) != 3:
    print('usage: %s data predictions' % sys.argv[0])
    sys.exit()

data_file = sys.argv[1]
predictions_file = sys.argv[2]

data = open(data_file)
predictions = open(predictions_file)

# Load the real labels.
if 'multitask' in data_file:

    # compare across tasks
    true_labels = []
    pertask_true_labels = []
    for line in data:
        if len(line.strip()) == 0:
            continue
        true_labels.append(line.split()[0])
        pertask_true_labels.append(list(line.split()[0]))

    predicted_labels = []
    pertask_predicted_labels = []
    for line in predictions:
        predicted_labels.append(line.strip())
        pertask_predicted_labels.append(list(line.strip()))

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    pertask_true_labels = np.array([a for b in true_labels for a in b])
    pertask_predicted_labels = np.array([a for b in predicted_labels for a in b])


elif 'semisup' in data_file:
    true_labels = []
    for line in data:
        if len(line.strip()) == 0:
            continue
        true_labels.append(line.split()[0])
    true_labels = np.array(true_labels)

    predicted_labels = []
    for line in predictions:
        predicted_labels.append(line.strip())
    predicted_labels = np.array(predicted_labels)

    ## only compare those are available
    data_available_idx = np.where(true_labels!='-1')[0]
    true_labels = true_labels[data_available_idx]
    predicted_labels = predicted_labels[data_available_idx]


data.close()
predictions.close()

if len(predicted_labels) != len(true_labels):
    print('Number of lines in two files do not match.')
    sys.exit()


correct_mask = true_labels == predicted_labels
num_correct = float(correct_mask.sum())
total = correct_mask.size
accuracy = num_correct / total

print('Accuracy: %f (%d/%d)' % (accuracy, num_correct, total))


if 'multitask' in data_file:

    correct_mask = pertask_true_labels == pertask_predicted_labels
    num_correct = float(correct_mask.sum())
    total = correct_mask.size
    accuracy = num_correct / total

    print('Multitask Accuracy across tasks: %f (%d/%d)' % (accuracy, num_correct, total))




