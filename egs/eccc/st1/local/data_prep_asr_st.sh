#!/bin/bash

# Copyright 2021 Kyoto University (Soky Kak for ECCC)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <dataset-path> <train_dir> <lang>"
  echo " $0 ~/ST data/local/ en km"
  exit 1;
fi

dataset_dir=$1
data_dir=$2
src_lang=$3
tgt_lang=$4
audio_dir=${dataset_dir}/${src_lang}

# data directory check
if [ ! -d $audio_dir ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi


# if [ ${tgt_lang} == ${src_lang} ]; then
src_text="${dataset_dir}/${src_lang}/text.${src_lang}.${src_lang}" #source text
# else
tgt_text="${dataset_dir}/${src_lang}/text.${src_lang}-${tgt_lang}.${tgt_lang}" #targets text
# fi

tmp_dir=${data_dir}/tmp
mkdir -p $tmp_dir
# find wav audio file for train, dev and test resp.
find $audio_dir/wav -iname "*.wav" > $tmp_dir/wav.flist.${src_lang}
find $audio_dir/wav_${src_lang}.${tgt_lang} -iname "*.wav" > $tmp_dir/wav.flist.${tgt_lang}

for lang in ${src_lang} ${tgt_lang}; do

  #clean transcription
  train_dir=${data_dir}/train.${src_lang}-${tgt_lang}.${lang}
  dev_dir=${data_dir}/dev.${src_lang}-${tgt_lang}.${lang}
  test_dir=${data_dir}/test.${src_lang}-${tgt_lang}.${lang}

  mkdir -p $train_dir
  mkdir -p $dev_dir
  mkdir -p $test_dir


  n=`cat $tmp_dir/wav.flist.${lang} | wc -l`
  echo data files found $n

  grep -i "train" $tmp_dir/wav.flist.${lang} > $train_dir/wav.flist || exit 1;
  grep -i "test" $tmp_dir/wav.flist.${lang} > $test_dir/wav.flist || exit 1;
  grep -i "dev" $tmp_dir/wav.flist.${lang} > $dev_dir/wav.flist || exit 1;
  #Transcriptions preparation
  for dir in $train_dir $dev_dir $test_dir; do
    echo "Preparing data in $dir."
    sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
    sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
    paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all

    #filter target utterances
    if [ ${lang} == ${src_lang} ]; then
      utils/filter_scp.pl -f 1 $dir/utt.list ${src_text} > $dir/transcripts
    else
      utils/filter_scp.pl -f 1 $dir/utt.list ${tgt_text} > $dir/transcripts
    fi
    awk '{print $1}' $dir/transcripts > $dir/utt.list
    #source utterance

    utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
    utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp

    sort -u $dir/transcripts > $dir/text

    utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

    # error check
    n=$(cat $dir/utt.list | wc -l)
    n_src=$(cat $dir/text | wc -l)
    # n_tgt=$(cat $dir/${tgt_lang}.text | wc -l)
    [ ${n} -ne ${n_src} ] && echo "Warning: expected ${n} data data files, source found ${n_src}" && exit 1;
    # [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data data files, target found ${n_tgt}" && exit 1;

    rm $dir/utt.list $dir/utt2spk_all $dir/wav.scp_all $dir/transcripts $dir/wav.flist

    #utils/validate_data_dir.sh --no-feats --no-text $dir
    utils/validate_data_dir.sh --no-feats $dir

    #text normalization after aligning with wav in each {set}
    #for lang in ${tgt_lang} ${src_lang}; do
    if [ ${lang} != "km" ]; then
      cut -d " " -f 1 $dir/text > ${dir}/${lang}.id
      cut -d " " -f 2- $dir/text > ${dir}/${lang}.org
      # normalize punctuation
      normalize-punctuation.perl -l ${lang} < ${dir}/${lang}.org > ${dir}/${lang}.norm

      # lowercasing
      lowercase.perl < ${dir}/${lang}.norm > ${dir}/${lang}.norm.lc
      cp ${dir}/${lang}.norm ${dir}/${lang}.norm.tc

      # remove punctuation
      local/remove_punctuation.pl < ${dir}/${lang}.norm.lc > ${dir}/${lang}.norm.lc.rm

      
      # tokenization
      tokenizer.perl -l ${lang} -q < ${dir}/${lang}.norm.tc > ${dir}/${lang}.norm.tc.tok
      tokenizer.perl -l ${lang} -q < ${dir}/${lang}.norm.lc > ${dir}/${lang}.norm.lc.tok
      tokenizer.perl -l ${lang} -q < ${dir}/${lang}.norm.lc.rm > ${dir}/${lang}.norm.lc.rm.tok

      paste -d " " ${dir}/${lang}.id ${dir}/${lang}.norm.tc.tok | sort > ${dir}/text.tc
      paste -d " " ${dir}/${lang}.id ${dir}/${lang}.norm.lc.tok | sort > ${dir}/text.lc
      paste -d " " ${dir}/${lang}.id ${dir}/${lang}.norm.lc.rm.tok | sort > ${dir}/text.lc.rm

      # save original and cleaned punctuation
      lowercase.perl < ${dir}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dir}/punctuation
      lowercase.perl < ${dir}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dir}/punctuation.clean
    else
      sort $dir/text > ${dir}/text.tc
      cp ${dir}/text.tc ${dir}/text.lc
      cp ${dir}/text.tc ${dir}/text.lc.rm
      cut -d " " -f 2- $dir/text > ${dir}/${lang}.org
      # save original and cleaned punctuation
      lowercase.perl < ${dir}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dir}/punctuation
    fi
  done
done

# for set in train dev test; do
#     # Copy stuff intoc its final locations [this has been moved from the format_data script]
#     mkdir -p ${data_dir}/${set}.${src_lang}-${tgt_lang}


#     utils/copy_data_dir.sh ${data_dir}/local/${src_lang}-${tgt_lang}/${set} ${data_dir}/${set}.${src_lang}-${tgt_lang}
#     #utils/copy_data_dir.sh --utt-suffix -${src_lang}-${tgt_lang} ${data_dir}/local/${src_lang}-${tgt_lang}/dev data/dt.${src_lang}-${tgt_lang}
#     #utils/copy_data_dir.sh --utt-suffix -${src_lang}-${tgt_lang} ${data_dir}/local/${src_lang}-${tgt_lang}/test data/et.${slang}-${tlang}

#     for l in ${src_lang} ${tgt_lang}; do
#         for case in tc lc lc.rm; do
#             cp ${data_dir}/local/${src_lang}-${tgt_lang}/${set}/text.${case}.${l} ${data_dir}/${set}.${src_lang}-${tgt_lang}/text.${case}.${l}
#         done
#     done
#     utils/fix_data_dir.sh --utt_extra_files \
#         "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" \
#         ${data_dir}/${set}.${src_lang}-${tgt_lang}

#     # error check
#     # n_seg=$(cat data/${set}.en-${tgt_lang}/segments | wc -l)
#     # n_text=$(cat data/${set}.en-${tgt_lang}/text.tc.${tgt_lang} | wc -l)
#     # [ ${n_seg} -ne ${n_text} ] && echo "Warning: expected ${n_seg} data data files, found ${n_text}" && exit 1;
# done

rm -r $tmp_dir

echo "$0: Data preparation is succeeded"
exit 0;
