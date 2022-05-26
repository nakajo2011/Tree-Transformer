import argparse
import graph
import os
from converter import SortIdConverter
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-no_cuda', action='store_true', help="Don't use GPUs.")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-pretrained_dir',default='pretrained',help='bert pretrained dir. using for tokenizer')
    parser.add_argument('-seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_step', type=int, default=100000, help='sequence length')
    parser.add_argument('-data_dir',default='data_dir',help='data dir')
    parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train', action='store_true',help='whether train the model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-graph', action='store_true',help='generate graph of test result tree')
    parser.add_argument('-graph_num', type=int, default=10, help='generate graph num')
    parser.add_argument('-valid_path',default='data_tv/valid.txt',help='validation data path')
    parser.add_argument('-train_path',default='data_tv/train.txt',help='training data path')
    parser.add_argument('-test_path',default='data_tv/test.txt',help='testing data path')
    parser.add_argument('-meta_data_path',default='data_tv/tv-meta-data.csv',help='tv meta data file path')

    parser.add_argument('-ff_vec', type=int, default=3072, help='intermediate layer vec size')
    parser.add_argument('-hidden_vec', type=int, default=256, help='hidden vec size')
    parser.add_argument('-encoder_layer_num', type=int, default=8, help='bert encoder layer num')
    parser.add_argument('-attention_heads', type=int, default=8, help='multi-attention-heads num')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)

    if args.graph:
        # 目視でわかりやすいようにsortidを番組情報に変換
        converter = SortIdConverter(model_dir=args.model_dir, tv_meta_data_file=args.meta_data_path)
        converted = converter.convert()

        # グラフ化
        graph.generate_images(args.model_dir, converted, args.graph_num)
    elif args.train:
        solver.train()
    elif args.test:
        solver.test()
