import numpy as np

from metrics.FVD.evaluate_FVD import get_embeddings
from metrics.DTFVD.DTFVD_Score import embedding_I3D, embedding_I3D_32, load_model


def compute_I3D_diversity(seq1, n_samples):
    """
    Computes diversity based on I3D backbone trained on kinetics

    Input: PyTorch tensor of shape (BatchSize, NumberSamples, Time, Channel, H, W)
        Important input needs to be in range [-1, 1] !

    """
    print('Evaluate Diversity score based on I3D trained on kinetics')

    input_shape = seq1.shape[0] // 16 * 16
    seq1 = seq1[:seq1.size(0) // 16 * 16].reshape(-1, 16, seq1.size(2), 3, seq1.size(-1), seq1.size(-1))

    assert seq1.min() < 0
    embed = get_embeddings(seq1)
    embed = embed.reshape(input_shape, n_samples, -1)
    div = []
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                div.append(((embed[:, i] - embed[:, j]) ** 2).mean())

    print(f'Diversity score of {np.mean(div)} using I3D kinetics backbone')


def compute_DTI3D_diversity(seq1):
    """
    Computes diversity based on I3D backbone trained on kinetics

    Input: PyTorch tensor of shape (BatchSize, NumberSamples, Time, Channel, H, W)
        Important input needs to be in range [-1, 1] !

    """
    ## load I3D
    length = seq1.size(2)
    I3D = load_model(length=32) if length > 16 else load_model(length=16)
    _ = I3D.cuda()

    assert seq1.min() < 0
    batch_size = 20
    seq1 = seq1.transpose(0, 1)
    embed = []
    for seq in seq1:
        embedding = embedding_I3D_32(I3D, seq, batch_size, True) if length > 16 else embedding_I3D(I3D, seq, batch_size, True)
        embed.append(embedding)
    embed = np.stack(embed, 1)
    div =[]
    for i in range(5):
        for j in range(5):
            if i != j:
                div.append(((embed[:, i]-embed[:, j])**2).mean())

    print(f'Diversity score of {np.mean(div)*1000} using I3D backbone pretrained on dynamic textures')
