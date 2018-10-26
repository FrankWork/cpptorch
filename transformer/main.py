from model import *

def test_pe():
	plt.figure(figsize=(15,5))
	pe = PositionalEncoding(20, 0)
	y = pe.forward(Variable(torch.zeros(1,100,20)))
	plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
	plt.legend(["dim %d" %p for p in [4,5,6,7]])

def data_gen(V, batch, nbatches, add_one=False):
	"Generate random data for a src-tgt copy task."
	if add_one:
		V -= 1
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
		data[:, 0] = 1
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		if add_one:
			tgt += 1
		yield Batch(src, tgt, 0)

def copy_task(add_one=False):
	'''
	A simple copy task
	'''
	V = 11
	criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
	model = make_model(V, V, N=2)
	adam = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, adam)

	print(model) # print model structure

	for epoch in range(10):
		print('Epoch: %d'%epoch)
		model.train() # set in train mode
		loss_fn = SimpleLossCompute(model.generator, criterion, model_opt)
		run_epoch(data_gen(V, 30, 20, add_one), model, loss_fn)
		model.eval() # set in eval mode
		loss_fn = SimpleLossCompute(model.generator, criterion, None)
		print(run_epoch(data_gen(V, 30, 5, add_one), model, loss_fn))
	
	model.eval()
	src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
	src_mask = Variable(torch.ones(1,1,10))
	print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

def test_labelsmoothing():
	crit = LabelSmoothing(5, 0, 0.1)
	def loss(x):
		d = x + 3 * 1
		predict = torch.FloatTensor([[0, x/d, 1/d, 1/d, 1/d],])
		predict = Variable(predict.log())
		target = Variable(torch.LongTensor([1]))
		#print(predict, target)
		return crit(predict, target).data[0]
	print([loss(x) for x in range(1,100)])
	#exit()

	crit = LabelSmoothing(5, 0, 0.4)
	predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
								 [0, 0.2, 0.7, 0.1, 0],
								 [0, 0.2, 0.7, 0.1, 0]])
	v = crit(Variable(predict.log()),
			 Variable(torch.LongTensor([2, 1, 0])))
	# plt.imshow(crit.true_dist)
	print(v)


print('hello world')
copy_task(add_one=False) # 擅长复制，非常不擅长加一
# test_labelsmoothing()

