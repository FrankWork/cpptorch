## extract wikipedia

```bash
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2  # 1.6G , 解压后6G
wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
python2 WikiExtractor.py -o zhwiki-xml zhwiki-latest-pages-articles.xml.bz2
du -h --max-depth 1 zhwiki-xml/ # 1.4G
python merge_files.py       # zhwiki.txt 1.1G

# 繁体转换为简体 
opencc -i zhwiki.txt -o wiki_zh_cn.txt -c ~/local/share/opencc/t2s.json

  
```

## train wordpiece 

```bash

###  按文本长度排序
# awk在每行前面加上长度信息，忽略长度小于10的文本
# 排序，
# 用sed删除长度信息
awk 'BEGIN { FS=RS } {if (length($0)>10) print length, $0}' wiki_zh_cn.txt |
sort +0n -1 | 
sed 's/^[0-9][0-9]* //' >  wiki.lines

wc -l wiki.lines  #3974989
cat wiki.lines | tqdm | uniq > wiki.lines.uniq
wc -l wiki.lines.uniq # 3901598

# 百度分词，太慢了
python infer.py --batch_size 5 --model_dir bd_lac_conf --input_path \
	wiki.lines.uniq --output_file wiki.lines.toks

# thulac
thulac -seg_only -model_dir models -input wiki.lines.uniq -output wiki.lines.toks
527秒 = 8分钟

bpe, word or char

# subword encoding
spm_train --input=wiki.lines.toks \
	--model_prefix=bpe_50k \
	--vocab_size=50000 \
	--character_coverage=0.9995 \
	--model_type=bpe \
	--input_sentence_size=3901598

spm_encode --model=bpe_50k.model --output_format=piece < input > output
```



