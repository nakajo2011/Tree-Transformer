import torch
import torch.nn as nn

from models import *
from parse import *
from utils import *


class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.no_cuda = args.no_cuda
        self.ff_vec = args.ff_vec # intermediate layer vec size
        self.hidden_vec = args.hidden_vec
        self.encoder_layer_num = args.encoder_layer_num
        self.attention_heads = args.attention_heads
        self.seq_length = args.seq_length

        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
        
        self.data_utils = data_utils(args)
        self.model = self._make_model(
            vocab_size=self.data_utils.vocab_size,
            N=self.encoder_layer_num,
            d_ff=self.ff_vec,
            d_model=self.hidden_vec,
            h=self.attention_heads,
            d_input=self.seq_length
        )

        self.test_vecs = None
        self.test_masked_lm_input = []

    def _make_model(self, vocab_size, N=10, 
            d_input=50, d_model=256, d_ff=2048, h=8, dropout=0.1):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
            group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout, d_input)
            word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
            model = Encoder(EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                    N, d_model, vocab_size, c(word_embed))
            
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            if self.no_cuda:
                model = model
            else:
                model = model.cuda()
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.bert = model

            return nn.DataParallel(model)


    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(f'{name}:{param.data.size()}')
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)

        data_yielder = self.data_utils.train_data_yielder()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        # optim = BertAdam(self.model.parameters(), lr=1e-4)
        
        total_loss = []
        start = time.time()
        total_step_time = 0.
        total_masked = 0.
        total_token = 0.
        total_data = len(self.data_utils.training_data)

        for step in range(self.args.num_step):
            self.model.train()
            batch = data_yielder.__next__()

            batch_size = len(batch["input"])
            step_start = time.time()
            out,break_probs = self.model.forward(batch['input'].long(), batch['input_mask'])
            
            loss = self.bert.masked_lm_loss(out, batch['target_vec'].long())
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
            optim.step()

            total_loss.append(loss.detach().cpu().numpy())

            total_step_time += time.time() - step_start
            
            if step % 200 == 1:
                elapsed = time.time() - start
                train_passed = step * batch_size
                epocs = train_passed / total_data
                print("Epoch Step: %d(%.4f epocs: %d/%d) Loss: %f Total Time: %f Step Time: %f" %
                        (step, epocs, train_passed, total_data, np.mean(total_loss), elapsed, total_step_time))
                self.model.train()
                print()
                start = time.time()
                total_loss = []
                total_step_time = 0.


            if step % 1000 == 0:
                print('saving!!!!')
                
                model_name = 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))


    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        if self.no_cuda:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['state_dict'])
        else:
            self.model.load_state_dict(torch.load(path)['state_dict'])

        self.model.eval()
        txts = get_test(self.args.test_path)

        vecs = [self.data_utils.text2id(txt, self.seq_length) for txt in txts]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs]
        self.test_vecs = cc(vecs, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts

        self.write_parse_tree()


    def write_parse_tree(self, threshold=0.8):
        batch_size = self.args.batch_size

        result_dir = os.path.join(self.model_dir, 'result/')
        make_save_dir(result_dir)
        f_b = open(os.path.join(result_dir,'brackets.json'),'w')
        f_t = open(os.path.join(result_dir,'tree.txt'),'w')
        f_d = os.path.join(result_dir, 'datas')
        make_save_dir(f_d)
        print(f'text vecs total: {self.test_vecs.shape}')

        for b_id in range(int(len(self.test_txts)/batch_size)+1):
            vecs = self.test_vecs[b_id*batch_size:(b_id+1)*batch_size]
            if len(vecs) == 0:
                print(f'vecs={len(vecs)}: vecs is empty, so break loop.')
                break

            out,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])
            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(self.test_txts[b_id*batch_size+i].strip().split())

                bp = get_break_prob(break_probs[i])[:,1:length]

                print(f'bp:{bp.shape}')
                text = self.test_txts[b_id*batch_size+i].strip()
                index = b_id*batch_size+i
                print(len(text.split()))
                data_json = {}
                data_json['tokens'] = text
                ca_data = {}
                for bpindex in range(bp.shape[0]):
                    ca_label = f'Layer{bpindex}'
                    ca_data[ca_label] = bp[bpindex].tolist()
                data_json['datas'] = ca_data
                json.dump(data_json, open(os.path.join(f_d, f'bp_{index}.json',), 'w'))

                model_out = build_tree(bp, bp.shape[0]-1, 0, length-1, threshold)
                if (0, length) in model_out:
                    model_out.remove((0, length))
                if length < 2:
                    model_out = set()
                f_b.write(json.dumps(list(model_out))+'\n')

                """
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                """

                nltk_tree = dump_tree(bp,  bp.shape[0]-1, 0, length-1, self.test_txts[b_id*batch_size+i].strip().split(), threshold)
                f_t.write(str(nltk_tree).replace('\n','').replace(' ','') + '\n')