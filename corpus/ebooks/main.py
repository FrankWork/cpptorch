#! /usr/bin/python3
import hashlib
import os
import re
import time
import argparse
import subprocess
from multiprocessing import Pool

def list_files_recursive(dir):
    for root, dirs, files in os.walk(dir):
        for name in files:
            real_path = os.path.join(root, name)
            # 'dir/subdir/file.mobi' -> '/subdir/file.mobi'
            #                 -> 'subdir/file.mobi'
            relative_path = re.sub(dir, '', real_path)[1:]
            yield real_path, relative_path

def get_file_md5(filename):
    m = hashlib.md5()

    with open(filename, 'rb') as file:
        while True:
            data = file.read(10240)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

def remove_duplicate_by_md5(base_dir):
    '''根据md5删除重复文件
    '''
    print(time.ctime())
    md5s = {}
    for real_path, relative_path in list_files_recursive(base_dir):
        md5 = get_file_md5(real_path)
        if md5 not in md5s:
            md5s[md5] = [real_path]#[relative_path]
        else:
            md5s[md5].append(real_path)#relative_path)
    print(time.ctime())

    with open('duplicate.txt', 'w') as f:
        for key in md5s.keys():
            files = md5s[key]
            if len(files)>1:
                files = sorted(files)[:-1]
                for file in files:
                    os.remove(file)
                    f.write(file+'\n')

def list_files(base_dir):
    '''输出所有文件
    '''
    for _, relative_path in list_files_recursive(base_dir):
        #print(os.path.basename(relative_path))
        print(relative_path)

def meta_info(base_dir):
    '''输出所有ebook-meta, 有些meta信息非常不可靠
    '''
    for real_path, relative_path in list_files_recursive(base_dir):
        out_bytes = subprocess.check_output(['ebook-meta', real_path])
        print(relative_path)
        print(out_bytes.decode('utf-8'))
        print()

def convert_fn(paths):
    real_path, out_path = paths
    try:
        subprocess.call(['ebook-convert', real_path, out_path])
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)

def convert_files(base_dir, out_dir):
    ''' .mobi .pdf .txt .epub -> .txt
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 生成所有路径
    all_files = []
    files_set = set()
    for real_path, relative_path in list_files_recursive(base_dir):
        basename = os.path.basename(relative_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(out_dir, name+'.txt')
        i = 1
        while out_path in files_set:
            i += 1
            out_path = os.path.join(out_dir, '%s_dup%d.txt'%(name, i))
        files_set.add(out_path)
        all_files.append( (real_path, out_path) )

    # 多进程处理
    pool = Pool(8)
    pool.map(convert_fn, all_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="list", help='list files or remove duplicate files')
    parser.add_argument('--base_dir', default="/mnt/d/lzh/电子书/mobi", help='')
    parser.add_argument('--out_dir', default="txt-books", help='')
    args = parser.parse_args()

    if args.mode == 'list':
        list_files(args.base_dir)
    elif args.mode == 'remove':
        remove_duplicate_by_md5(args.base_dir)
    elif args.mode == 'meta':
        meta_info(args.base_dir)
    elif args.mode == 'convert':
        convert_files(args.base_dir, args.out_dir)

