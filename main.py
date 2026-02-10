from models.model import *
from utils.data_util import load_data
from utils.data_loader import *
import numpy as np
import argparse
import torch
import time
import os
import pickle
import torch.nn.functional as F


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 4000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'BM3LP',   
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'BriM',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 512,
        'save': 1
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

# 保持原逻辑
if args.model in ['ConvE', 'TuckER', 'Mutan', 'BM3LP']:
    corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)
else:
    corpus = ConvKBCorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)

args.entity2id = entity2id
args.relation2id = relation2id


# =========================
# =========================
model_name = {
    'ConvE': ConvE,
    'TuckER': TuckER,
    'Mutan': Mutan,
    'BM3LP': BM3LP,
}


def train_encoder(args):
    model = DFS_RGAT(args)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=float(args.gamma))

    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)

    corpus.batch_size = len(corpus.train_triples)
    corpus.neg_num = 2

    best_loss = float('inf')

    for epoch in range(args.epochs_gat):
        model.train()
        np.random.shuffle(corpus.train_triples)
        train_indices, _ = corpus.get_batch(0)
        train_indices = torch.LongTensor(train_indices).to(args.device)

        optimizer.zero_grad()
        entity_embed, relation_embed = model.forward(corpus.train_adj_matrix, train_indices)
        loss = model.loss_func(train_indices, entity_embed, relation_embed)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print("Epoch {} , loss {:.4f}".format(epoch, loss.item()))

        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(f'./checkpoint/{args.dataset}', exist_ok=True)
            torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/DFS_RGAT_best.pth')
            print(f"🔥 Save best encoder, loss={best_loss:.4f}")


def train_decoder(args):
    if args.encoder:
        model_gat = DFS_RGAT(args)

    model = model_name[args.model](args)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    if args.cuda is not None and int(args.cuda) >= 0:
        if args.encoder:
            model_gat = model_gat.to(args.device)
            model_gat.load_state_dict(
                torch.load(f'./checkpoint/{args.dataset}/DFS_RGAT_best.pth'), strict=False)
        model = model.to(args.device)

    t_total = time.time()
    best_val_mrr = 0
    best_test_mrr = 0
    best_epoch = 0
    best_test_epoch = 0
    best_val_metrics = None
    best_test_metrics = None

    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        corpus.shuffle()

        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices).to(args.device)
            train_values = train_values.to(args.device)

            head = train_indices[:, 0]
            outputs = model.forward(train_indices)
            loss = model.loss_func(outputs, train_values, head)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print("\nEpoch {:04d}, avg loss {:.4f}".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss)))

            model.eval()
            with torch.no_grad():
                val_metrics = corpus.get_validation_pred(model, 'val')
                test_metrics = corpus.get_validation_pred(model, 'test')

            print("[VAL ]", model.format_metrics(val_metrics, 'val'))
            print("[TEST]", model.format_metrics(test_metrics, 'test'))

            if val_metrics['Mean Reciprocal Rank'] > best_val_mrr:
                best_val_mrr = val_metrics['Mean Reciprocal Rank']
                best_epoch = epoch
                best_val_metrics = val_metrics
                torch.save(model.state_dict(),
                           f'./checkpoint/{args.dataset}/{args.model}_best.pth')
                print(f"🔥 Save best model at epoch {epoch+1}, val MRR={best_val_mrr:.4f}")

            if test_metrics['Mean Reciprocal Rank'] > best_test_mrr:
                best_test_mrr = test_metrics['Mean Reciprocal Rank']
                best_test_epoch = epoch
                best_test_metrics = test_metrics

    print('\nOptimization Finished!')
    print('Total time: {:.2f}s'.format(time.time() - t_total))

    print(f"\nLoading best model from epoch {best_epoch+1}")
    model.load_state_dict(
        torch.load(f'./checkpoint/{args.dataset}/{args.model}_best.pth')
    )
    model.eval()
    with torch.no_grad():
        final_test_metrics = corpus.get_validation_pred(model, 'test')

    print("\n================ FINAL RESULTS ================")
    print(f"Best VAL MRR : {best_val_mrr:.4f} at epoch {best_epoch+1}")
    print(f"Best TEST MRR (during training) : {best_test_mrr:.4f} at epoch {best_test_epoch+1}")

    print("\nFinal VAL (best val model):")
    print(model.format_metrics(best_val_metrics, 'val'))

    print("\nBest TEST (during training):")
    print(model.format_metrics(best_test_metrics, 'test'))

    print("\nFinal TEST (best val model):")
    print(model.format_metrics(final_test_metrics, 'test'))
    print("==============================================")

    if args.save:
        torch.save(model.state_dict(),
                   f'./checkpoint/{args.dataset}/{args.model}_final.pth')
        print('Saved final model!')

    save_embeddings(model, args.dataset)


def save_embeddings(model, dataset_name, save_path='./embeddings'):
    dataset_name = dataset_name.replace('/', '_')
    os.makedirs(save_path, exist_ok=True)

    structure_entity_embeddings = model.entity_embeddings.weight.detach().cpu().numpy()
    structure_relation_embeddings = model.relation_embeddings.weight.detach().cpu().numpy()
    pickle.dump(structure_entity_embeddings, open(os.path.join(save_path, f'{dataset_name}_structure_entity_embeddings.pkl'), 'wb'))
    pickle.dump(structure_relation_embeddings, open(os.path.join(save_path, f'{dataset_name}_structure_relation_embeddings.pkl'), 'wb'))

    image_entity_embeddings = model.img_entity_embeddings.weight.detach().cpu().numpy()
    image_relation_embeddings = model.img_relation_embeddings.weight.detach().cpu().numpy()
    pickle.dump(image_entity_embeddings, open(os.path.join(save_path, f'{dataset_name}_image_entity_embeddings.pkl'), 'wb'))
    pickle.dump(image_relation_embeddings, open(os.path.join(save_path, f'{dataset_name}_image_relation_embeddings.pkl'), 'wb'))

    text_entity_embeddings = model.txt_entity_embeddings.weight.detach().cpu().numpy()
    text_relation_embeddings = model.txt_relation_embeddings.weight.detach().cpu().numpy()
    pickle.dump(text_entity_embeddings, open(os.path.join(save_path, f'{dataset_name}_text_entity_embeddings.pkl'), 'wb'))
    pickle.dump(text_relation_embeddings, open(os.path.join(save_path, f'{dataset_name}_text_relation_embeddings.pkl'), 'wb'))


if __name__ == '__main__':
    # train_encoder(args)
    train_decoder(args)