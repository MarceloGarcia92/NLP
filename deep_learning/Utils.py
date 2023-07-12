from numpy import array

def text_numpy(sequence):
    return sequence.numpy()

def text_numpy_decode(sequence):
    return text_numpy(sequence).decode('utf8')

def preprocesing_tensors(tensor_set):
    list_seq = list()
    list_label = list()

    for seq, label in tensor_set:
        list_seq.append(text_numpy_decode(seq))
        list_label.append(text_numpy(label))

    arr_seq = array(list_seq)
    arr_label = array(list_label)

    return arr_seq, arr_label