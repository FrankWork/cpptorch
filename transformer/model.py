# http://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many other
	models
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		x = self.encode(src, src_mask)
		return self.decode(x, src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		x = self.src_embed(src)
		return self.encoder(x, src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		y = self.tgt_embed(tgt)
		return self.decoder(y, memory, src_mask, tgt_mask)

class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)
	
	def forward(self, x):
		x = self.proj(x)
		return F.log_softmax(x, dim=-1)

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
	"Core encoder is a stack of N layers"
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.layernorm = LayerNorm(layer.size)
	
	def forward(self, x, mask):
		"Pass the input (and mask) through each layer in turn."
		for layer in self.layers:
			x = layer(x, mask)
		return self.layernorm(x)

class LayerNorm(nn.Module):
	"Construct a layernorm module."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.w_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps
	
	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.w_2 * (x-mean) / (std+self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.layernorm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		x_norm = self.layernorm(x)
		x_layer = sublayer(x_norm)
		x_drop = self.dropout(x_layer)
		return x + x_drop 

class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward."
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
                # self_attn + feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout),2)
		self.size = size
	
	def forward(self, x, mask):
		"Follow Vaswani's paper Figure 1 (left) for connection."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.layernorm = LayerNorm(layer.size)
	
	def forward(self, x, memory, src_mask, tgt_mask): 
		# x is target, encoder result is memory
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.layernorm(x)

class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward."
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout),3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m=memory
		x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,tgt_mask))
		x = self.sublayer[1](x, lambda x:self.src_attn(x,m,m,src_mask))
		return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
	return torch.from_numpy(mask)==0

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])


def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2,-1)) 
	scores = scores / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask==0,-1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, n_heads, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % n_heads == 0
		self.d_k = d_model // n_heads
		self.h = n_heads
		self.linears = clones(nn.Linear(d_model, d_model),4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# same mask applied to all heads
			mask = mask.unsqueeze(1)
		n = query.size(0)

		# 1. Do all the linear projections in batch from d_model => h x d_k
		query, key, value = \
				[l(x).view(n, -1, self.h, self.d_k).transpose(1,2)
				 for l, x in zip(self.linears, (query, key, value))]
		# 2. Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask,
								 dropout=self.dropout)
		# 3. "concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous().view(n, -1, self.h*self.d_k)
		return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.l_1 = nn.Linear(d_model, d_ff)
		self.l_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		x = self.l_1(x)
		x = F.relu(x)
		x = self.dropout(x)
		return self.l_2(x)

class Embeddings(nn.Module):
	def __init__(self, d_model, vocab): # int, int
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)# lookup table
		self.d_model = d_model
	
	def forward(self, x):
		return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0., max_len).unsqueeze(1) # shape (max_len, 1)
		div_term = torch.exp(torch.arange(0., d_model, 2) * 
				-(math.log(10000.0)/d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe) # no update during training
	
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],
				requires_grad=False)
		return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8,
		dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))

	# important!
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model

# tmp_model = make_model(10, 10, 2)

class Batch:
	"Object for holding a batch of data with mask during training."
	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum()
	
	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt!=pad).unsqueeze(-2)
		sm = Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		tgt_mask = tgt_mask & sm
		return tgt_mask

def run_epoch(data_iter, model, loss_compute):
	"Standard Training and Logging Function"
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0

	for i, batch in enumerate(data_iter):
		out = model.forward(batch.src, batch.trg,
							batch.src_mask, batch.trg_mask)
		loss = loss_compute(out, batch.trg_y, batch.ntokens)

		total_loss += loss
		total_tokens += batch.ntokens
		tokens += batch.ntokens
		if i%50 == 1:
			elapsed = time.time() - start
			print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
					(i, loss/batch.ntokens.float(), tokens.float()/elapsed))
			start = time.time()
			tokens = 0
	return total_loss / total_tokens.float()

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
	"Keep augmenting batch and calculate total number of tokens + padding."
	global max_src_in_batch, max_tgt_in_batch
	if count == 1:
		max_src_in_batch = 0
		max_tgt_in_batch = 0
	max_src_in_batch = max(max_src_in_batch, len(new.src))
	max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg)+2)
	src_elements = count * max_src_in_batch
	tgt_elements = count * max_tgt_in_batch
	return max(src_elements, tgt_elements)

class NoamOpt:
	"Optim wrapper that implements rate."
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0
	
	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()
	
	def rate(self, step=None):
		if step is None:
			step = self._step
		return self.factor * (self.model_size ** (-0.5) * \
			  min(step**(-0.5), step*self.warmup**(-1.5)))
	

def get_std_opt(model):
	adam = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98),
			eps=1e-9)
	return NoamOpt(model.src_embed[0].d_model, 2, 4000, adam)


# opts = [NoamOpt(512, 1, 4000, None),
#		NoamOpt(512, 1, 8000, None),
#		NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1,
#	20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])

# Regularization

class LabelSmoothing(nn.Module):
	'''
	https://arxiv.org/abs/1512.00567
	'''
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None

	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size-2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx]=0
		mask = torch.nonzero(target.data == self.padding_idx)
		
		if len(mask) > 0: #mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
	"A simple loss compute and train function."
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt
	
	def __call__(self, x, y, norm):
		x = self.generator(x)
		#print(x.dtype, y.dtype, norm.dtype)
		norm = norm.float()
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
							  y.contiguous().view(-1)) / norm
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		return loss.item()*norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
	for i in range(max_len-1):
		out = model.decode(memory, src_mask, 
						   Variable(ys),
						   Variable(subsequent_mask(ys.size(1))
							   .type_as(src.data)))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.data[0]
		ys = torch.cat([ys,
			torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=1)
	return ys



