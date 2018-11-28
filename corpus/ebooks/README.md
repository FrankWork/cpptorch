## md5重复文件

```bash
$ python3 main.py --mode remove 
Sun Nov 25 20:16:12 2018
Sun Nov 25 20:16:55 2018
```

## 编码转换

```bash
iconv -f gbk -t utf-8 infile.txt > outfile.txt
```

## 格式转换

```bash
sudo apt install calibre
# mobi -> txt
ebook-convert in_file.mobi out_file.txt 
# epub -> txt
ebook-convert in_file.epub out_file.txt

```

## 文件元信息

```bash
# 读取元信息, 有些信息非常不可靠
$ ebook-meta file.mobi
	Title               : 电子书名
	Author(s)           : 某某
	Publisher           : 某某出版社
	Languages           : zho
	Published           : 2009-07-30T16:00:00+00:00
	Identifiers         : mobi-asin:XXXXXXXXXX

# 设置元信息
$ ebook-meta --title "乔布斯传" file.mobi
```
