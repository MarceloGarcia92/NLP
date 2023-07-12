import io
import matplotlib.pyplot as plt

def embedding_projector(vocab_size, tokenizer, embeding_weights):
    reverse_word_idx = tokenizer.index_word

    out_vector = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_meta = io.open('meta.tsv', 'w', encoding='utf-8')

    for word_n in range(1, vocab_size):
        word_name = reverse_word_idx[word_n]
        word_embeding = embeding_weights[word_n]

        out_meta.write(f'{word_name}\n')
        out_vector.write('\t'.join([str(x) for x in word_embeding]) + '\n')

    out_meta.close()
    out_vector.close()

def acc_loss_plots(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(4,4))
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(epochs, acc, 'bo', label='Training acc')
    ax[0].plot(epochs, val_acc, 'b', label='Validation acc')

    ax[1].plot(epochs, loss, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    fig.suptitle('Training and validation metrics')
    fig.legend()

    plt.show()