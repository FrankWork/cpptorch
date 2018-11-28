import re
import argparse
import glob
import time


re_newline = re.compile('\n+')
re_brace = re.compile('（[,，]*）')
filter_list = ['File:', 'Category:', 'Topic:', 'MediaWiki:', 'Wikipedia:']

def get_files(in_dir):
    for file in glob.glob(in_dir+'/*/wiki_*'):
        yield file

def get_docs(in_file):
    buf = ''
    with open(in_file) as f:
        for line in f:
            if line.startswith('<doc'):
                continue
            if line.startswith("</doc>"):
                doc = clean_text(buf)
                doc = filter_doc(doc)
                if doc:
                    yield doc
                buf = ''
                continue
            buf += line

def remove_title(doc):
    '''remove first line'''
    lines = [line for line in doc.split('\n') if line]
    lines = lines[1:]

    return '\n'.join(lines) + '\n'

def filter_doc(doc):
    for pattern in filter_list:
        if doc.startswith(pattern):
            return None

    doc = remove_title(doc)
    if len(doc) < 20:
        return None

    return doc
    
def clean_text(doc):
    doc = doc.strip() + '\n'
    doc = re_newline.sub('\n', doc)
    doc = re_brace.sub('', doc)
    return doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='zhwiki-xml', type=str, help='')
    parser.add_argument('--out_file', default='zhwiki.txt', type=str, help='')
    args = parser.parse_args()

    n_doc=0
    print(time.ctime())
    with open(args.out_file, 'w') as f:
        for file in get_files(args.in_dir):
           for doc in get_docs(file):
               f.write(doc)
               f.write('\n')
               n_doc += 1
               if n_doc % 100000 == 0:
                   print('n_doc %d' % n_doc, time.ctime())
    print(n_doc, time.ctime())

