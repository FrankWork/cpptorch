## wikipedia

```bash
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2  # 1.6G , 解压后6G
wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
python2 WikiExtractor.py -o zhwiki-xml zhwiki-latest-pages-articles.xml.bz2
du -h --max-depth 1 zhwiki-xml/ # 1.4G
python merge_files.py       # zhwiki.txt 1.1G

# 繁体转换为简体 
opencc -i zhwiki.txt -o wiki_zh_cn.txt -c ~/local/share/opencc/t2s.json

# subword encoding
spm_train --input=wiki_zh_cn.txt \
	--model_prefix=bpe_50k \
	--vocab_size=50000 \
	--character_coverage=0.9995 \
	--model_type=bpe \
	--input_sentence_size=5202412  
```

	




