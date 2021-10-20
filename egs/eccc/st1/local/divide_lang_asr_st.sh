#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
# Modified by Kak Soky, 2021 for Join ASR and ST only
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <set> <slang> <tlang> <data>>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1
src_lang=$2
tgt_lang=$3
data_dir=$4

# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p ${data_dir}/${set}.${src_lang}
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f ${data_dir}/${set}/${f} ]; then
        sort ${data_dir}/${set}/${f} > ${data_dir}/${set}.${src_lang}/${f}
    fi
done
sort ${data_dir}/${set}/text.lc.rm.${src_lang} > ${data_dir}/${set}.${src_lang}/text  # dummy
sort ${data_dir}/${set}/text.tc.${src_lang} > ${data_dir}/${set}.${src_lang}/text.tc
sort ${data_dir}/${set}/text.lc.${src_lang} > ${data_dir}/${set}.${src_lang}/text.lc
sort ${data_dir}/${set}/text.lc.rm.${src_lang} > ${data_dir}/${set}.${src_lang}/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" ${data_dir}/${set}.${src_lang}
if [ -f ${data_dir}/${set}.${src_lang}/feats.scp ]; then
    utils/validate_data_dir.sh ${data_dir}/${set}.${src_lang} || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav ${data_dir}/${set}.${src_lang} || exit 1;
fi

# for target language
mkdir -p ${data_dir}/${set}.${tgt_lang}
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f ${data_dir}/${set}/${f} ]; then
        sort ${data_dir}/${set}/${f} > ${data_dir}/${set}.${tgt_lang}/${f}
    fi
done
sort ${data_dir}/${set}/text.tc.${tgt_lang} > ${data_dir}/${set}.${tgt_lang}/text  # dummy
sort ${data_dir}/${set}/text.tc.${tgt_lang} > ${data_dir}/${set}.${tgt_lang}/text.tc
sort ${data_dir}/${set}/text.lc.${tgt_lang} > ${data_dir}/${set}.${tgt_lang}/text.lc
sort ${data_dir}/${set}/text.lc.rm.${tgt_lang} > ${data_dir}/${set}.${tgt_lang}/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" ${data_dir}/${set}.${tgt_lang}
if [ -f ${data_dir}/${set}.${tgt_lang}/feats.scp ]; then
    utils/validate_data_dir.sh ${data_dir}/${set}.${tgt_lang} || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav ${data_dir}/${set}.${tgt_lang} || exit 1;
fi
