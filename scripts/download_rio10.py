#!/usr/bin/env python
# Downloads RIO10 public data release
# Run with ./download.py
# -*- coding: utf-8 -*-

import sys
import argparse
import os

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib
import tempfile
import re

BASE_URL = 'http://campar.in.tum.de/public_datasets/RIO10/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/public_datasets/RIO10/RIO10TOU.pdf'
FILETYPES = ['seq', 'models', 'semantics', 'kapture']

# Alternatively, you can download sequence, semantic and model data via the command line:

# wget http://campar.in.tum.de/public_datasets/RIO10/Dataset/[seq|models]<scene_id>.zip e.g.:
# wget http://campar.in.tum.de/public_datasets/RIO10/Dataset/seq01.zip
# wget http://campar.in.tum.de/public_datasets/RIO10/Dataset/models04.zip
# wget http://campar.in.tum.de/public_datasets/RIO10/Dataset/semantics05.zip

def download_release(out_dir, file_types):
    print('Downloading RIO10 release to ' + out_dir + '...')
    for scene_id in range(1, 11): # 1-10
        download_scan(scene_id, out_dir, file_types)
    print('Downloaded RIO10 release.')


def download_file(url, out_file):
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.urlretrieve(url, out_file_tmp) 
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_scan(scene_id, out_dir, file_types):
    scene_str = 'scene' + str(scene_id).zfill(2)
    out_scene_dir = os.path.join(out_dir, scene_str)
    print('Downloading RIO10', scene_id, '...', out_dir, file_types, out_scene_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        if ft == 'kapture':
            for file_type in ['mapping', 'testing', 'validation']:
                url = DATA_URL + '/kapture/RIO10_scene' + str(scene_id).zfill(2) + '_' + file_type + '.tar.gz'
                out_file = out_scene_dir + '/RIO10_scene' + str(scene_id).zfill(2) + '_' + file_type + '.tar.gz'
                download_file(url, out_file)
        else:
            url = DATA_URL + '/' + ft + str(scene_id).zfill(2) + '.zip'
            out_file = out_scene_dir + '/' + ft + str(scene_id).zfill(2) + '.zip'
            download_file(url, out_file)


def get_filename(out_dir, scene_id):
    return os.path.join(out_dir, 'scene'+str(scene_id).zfill(2))


def main():
    parser = argparse.ArgumentParser(description='Downloads RIO10 public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', type=int, help='specific scene id to download [1-10]')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the RIO10 terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = input('')

    file_types = ['seq', 'models', 'semantics']

    if args.type:  # download file type
        file_type = args.type
        if file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = [file_type]
    if args.id:  # download single scene
        scene_id = int(args.id)
        if scene_id > 10 and scene_id <= 0:
            print('ERROR: Invalid scan id: ' + scene_id)
        else:
            if scene_id <= 10 and scene_id > 0:
                download_scan(scene_id, args.out_dir, file_types)
    else: # download entire release
        if len(file_types) == len(FILETYPES):
            print('Downloading the entire RIO10 release.')
        else:
            print('Downloading all RIO10 scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')

        download_release(args.out_dir, file_types)


if __name__ == '__main__': main()