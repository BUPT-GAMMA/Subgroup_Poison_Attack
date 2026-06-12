import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--attack', nargs='?', default='ours', help="Specify a attack method")
    parser.add_argument('--defense', default='Nodefense', choices=['Nodefense', 'Normbound', 'Bulyan', 'TrimmedMean', 'Krum', 'MultiKrum', 'Median', 'Pieck'])
    parser.add_argument('--dim', type=int, default=8, help='Dim of latent vectors.')
    parser.add_argument('--layers', nargs='?', default='[8,8]', help="Dim of mlp layers.")
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    parser.add_argument('--path', nargs='?', default='./data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ML-100K', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cuda:2' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')#0.001
    parser.add_argument('--generator_lr', type=float, default=0.001, help='generator learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs.')
    parser.add_argument('--generator_epochs', type=int, default=30, help='Number of training epochs of generator.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--items_limit', type=int, default=30, help='Limit of items.')
    parser.add_argument('--grad_limit', type=float, default=0.5, help='Limit of l2-norm of item gradients.')
    '''pieck defense'''
    parser.add_argument('--regula_size', type=int, default=10, help='Regula size.')#10 pieck defense
    parser.add_argument('--ipe_mu', type=float, default=4.5e+1, help='Defense.')#pieck defense
    parser.add_argument('--uea_mu', type=float, default=4.5e+1, help='Defense.')#pieck defense

    '''for saving files'''
    parser.add_argument('--mode', type=str, default='test', help='choose a model.')
    '''ablation study'''
    parser.add_argument('--is_cluster', action='store_false', help='whether to use cluster.')
    parser.add_argument('--is_distance', action='store_false', help='whether to use distance.')
    parser.add_argument('--is_adaptive_coef', action='store_false', help='whether to use adaptive coef.')
    '''paramters analysis'''
    parser.add_argument('--clients_limit', type=float, default=0.002, help='Limit of proportion of malicious clients.')
    parser.add_argument('--similarity_coef', type=float, default=0.1, help='Similarity Coefficient.')
    parser.add_argument('--distance_margin', type=int, default=15, help='distance_margin.')
    parser.add_argument('--cluster_num', type=int, default=30, help='cluster_num.')
    parser.add_argument('--epoch_coef', type=float, default=0.0005, help='epoch coefficient.')
    '''interested items analysis'''
    parser.add_argument('--interested_min_freq', type=float, default=0.2, help='Minimum frequency of interested items.')
    parser.add_argument('--interested_max_freq', type=float, default=1, help='Maximum frequency of interested items.')
    parser.add_argument('--num_interested_items', type=int, default=10, help='Number of interested items.')

    '''visualize emb'''
    parser.add_argument('--is_visualize', action='store_false', help='whether to visualize emb.')

    '''ldp param'''
    parser.add_argument('--laplace_lambda', type=float, default=9, help='laplace_lambda.')
    parser.add_argument('--clip', type=float, default=1, help='clip.')




    return parser.parse_args()


args = parse_args()


GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'ML': (512, 8, 3, 2, 64),
}
