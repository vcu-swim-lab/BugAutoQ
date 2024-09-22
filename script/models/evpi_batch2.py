import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import spatial
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import dataset_batch as dataset


class EvpiModel(nn.Module):

    def __init__(self, weights, hidden_dim=256):
        super(EvpiModel, self).__init__()
        self.hidden_dim = hidden_dim
        # layer 1 - embeddings
        weights = torch.FloatTensor(weights)
        self.emb_layer = nn.Embedding.from_pretrained(weights)
        self.emb_layer.requires_grad = False

        # layer 2 - LSTMs
        self.p_lstm = nn.LSTM(input_size=self.emb_layer.embedding_dim, hidden_size=hidden_dim, num_layers=5,
                              batch_first=True)
        # layer 3 - dense layer
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.emb_layer.embedding_dim)
        self.layer4 = nn.Linear(self.emb_layer.embedding_dim, self.emb_layer.embedding_dim)

    def forward(self, post, post_lengths):
        # sort data
        post, post_lengths, post_sorted_id = self.sort_batch(post, post_lengths)

        # process posts
        p_emb_out = self.emb_layer(post)
        p_emb_out = pack_padded_sequence(p_emb_out, post_lengths, batch_first=True)
        p_lstm_out, _ = self.p_lstm(p_emb_out)
        p_lstm_out, _ = pad_packed_sequence(p_lstm_out, batch_first=True)
        p_mean = p_lstm_out.mean(dim=1)

        # restore ordering
        p_mean, post_lengths = self.restore_ordering(p_mean, post_lengths, post_sorted_id)

        # linear layers
        x = F.relu(self.layer1(p_mean))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        answer_representation = self.layer4(x)
        return answer_representation

    def sort_batch(self, data, seq_len):
        """ Sort the data (B, T, D) and sequence lengths """
        sorted_seq_len, sorted_idx = seq_len.sort(0, descending=True)
        sorted_data = data[sorted_idx]
        return sorted_data, sorted_seq_len, sorted_idx

    def restore_ordering(self, data, seq_len, sorted_idx):
        """ Restore original ordering of data based on sorted ids """
        sorted_sorted_id, initial_id = sorted_idx.sort(0, descending=False)
        sorted_data = data[initial_id]
        sorted_len = seq_len[initial_id]
        return sorted_data, sorted_len


def get_device(cuda, cuda_no):
    if cuda is True:
        device = torch.device('cuda:'+str(cuda_no) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def compute_qa_cap(questions, answers, w2v_model):
    answers = answers.numpy()
    questions = questions.numpy()
    a_cap = np.ndarray((answers.shape[0], w2v_model.vectors.shape[1]))
    for idx, (q, a) in enumerate(zip(questions,answers)):
        a_words = [w2v_model.index2word[index] for index in a]
        q_words = [w2v_model.index2word[index] for index in q]
        a_cap[idx] = np.mean(np.concatenate((w2v_model[a_words],w2v_model[q_words])), axis=0)
    return torch.tensor(a_cap, dtype=torch.float)


# def compute_a_cap(answer, w2v_model):
#     answer = answer.numpy()
#     a_cap = np.ndarray((answer.shape[0], w2v_model.vectors.shape[1]))
#     for idx, tensor in enumerate(answer):
#         words = [w2v_model.index2word[index] for index in tensor]
#         a_cap[idx] = np.mean(w2v_model[words], axis=0)
#     return torch.tensor(a_cap, dtype=torch.float)


def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def run_evaluation(net, device, w2v_model, test_loader):
    logging.info('Running evaluation...')
    results = {}
    with torch.no_grad():
        for data in test_loader:
            answers = data['answer']
            questions = data['question']
            qa_cap = compute_qa_cap(questions, answers, w2v_model).numpy()

            if device.type != 'cpu':
                posts, post_len, answers = data['post'].to(device), data['post_len'].to(device), \
                                                             data['answer'].to(device)
            else:
                posts = data['post']
                post_len = data['post_len']

            postids = data['postid']
            utility = data['utility'].numpy()
            posts_origin = data['post_origin']
            answers_origin = data['answer_origin']
            questions_origin = data['question_origin']
            labels = data['label']

            outputs = net(posts, post_len)

            if device.type != 'cpu':
                outputs = outputs.cpu()

            outputs = outputs.numpy()

            for idx in range(0, len(postids)):
                postid = postids[idx]
                cos_sim = cosine_similarity(qa_cap[idx], outputs[idx])
                score = cos_sim * utility[idx]
                print('postid={4}\tcosine sim={0}\tutility={1}\tscore={2}\tlabel={3}'.format(cos_sim, utility[idx], score,
                                                                                 labels[idx], postid))

                if postid not in results:
                    results[postid] = (posts_origin[idx], list(), list())
                results[postid][1].append((score, questions_origin[idx], answers_origin[idx]))
                if labels[idx] == 1:
                    results[postid][2].extend([questions_origin[idx], answers_origin[idx]])

    return results


def evpi(cuda, cuda_no, w2v_model, args):
    device = get_device(cuda, cuda_no)
    logging.info('Running on {0}'.format(device))

    net = EvpiModel(w2v_model.vectors)
    net.to(device)

    train_loader, test_loader = dataset.get_datasets(w2v_model.vocab, args)

    loss_function = torch.nn.CosineEmbeddingLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    print(net)

    for epoch in range(args.n_epochs):
        logging.info('Epoch {0}/{1}'.format((epoch + 1), args.n_epochs))
        loss_sum = 0.0
        for data in train_loader:
            # compute a_cap and send it to device so it can be used for back propagation
            answers = data['answer']
            questions = data['question']
            qa_cap = compute_qa_cap(questions, answers, w2v_model)
            labels = data['label']

            if device.type != 'cpu':
                posts, post_len, qa_cap, labels = data['post'].to(device), data['post_len'].to(device), \
                                                                   qa_cap.to(device), labels.to(device)
            else:
                posts = data['post']
                post_len = data['post_len']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(posts, post_len)

            loss = loss_function(outputs, qa_cap, labels)
            loss.backward()

            optimizer.step()
            loss_sum += loss.item()

        logging.info('Loss: {0}'.format(loss_sum / len(train_loader)))

    results = run_evaluation(net, device, w2v_model, test_loader)
    return results

