#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

########################################################################################################################
#                                                    NOTE                                                              #
# This script is similar to run.sh, but I just modified to be able to support multiple token (word, subword, character)#
#                                                                                                                      #
########################################################################################################################
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16           # number of parallel jobs for decoding
debugmode=1
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

# preprocessing related
case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
eccc=/home/soky/ECCC/wav/ST #raw wav and text
data_dir=data # /home/soky/work1/data/trilingual #clean fbank, text
dumpdir=dump    # directory to dump full features
# target language related
src_lang=km
tgt_lang=en

# pre-training related
asr_model= #--asr_model ../asr1/exp/train_sp.en-fr.en_lc.rm_pytorch_train_bpe8000_specaug/results/model.val5.avg.best
mt_model=
enc_init=


# you can choose from de, es, fr, it, nl, pt, ro, ru
# if you want to train the multilingual model, segment languages with _ as follows:
# e.g., tgt_lang="de_es_fr"
# if you want to use all languages, set tgt_lang="all"

# use the same dict as in the ST task
use_st_dict=true

# bpemode (unigram or bpe)
nbpe=5000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.${src_lang}-${tgt_lang}.${src_lang}
train_set_prefix=train_sp
train_dev=dev.${src_lang}-${tgt_lang}.${src_lang}
recog_set=""
for lang in $(echo ${tgt_lang} | tr '_' ' '); do
    recog_set="${recog_set} test.${src_lang}-${lang}.${src_lang}"
done

# 1.1. data preparation
# 1.2. devide language
# 2.1. extract fbank
# 2.2. speech perturbation
# 2.2. align source1, source2, target1, target2(optional)
# 2.3. dump feature
# 2.4. update target
# 3. generate dictionary and json data.


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage 0: Data Preparation"
    for slang in $(echo ${src_lang} | tr '_' ' '); do
        for tlang in $(echo ${tgt_lang} | tr '_' ' '); do
            if [ ${slang} != ${tlang} ]; then
                local/data_prep_asr_st.sh ${eccc} ${data_dir} ${slang} ${tlang}
            fi
        done
    done
fi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    
    echo "stage 1: Feature Generation"
    fbankdir=${data_dir}/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for lang in ${tgt_lang} ${src_lang}; do
        for x in train.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang} test.${src_lang}-${tgt_lang}; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
                ${data_dir}/${x}.${lang} exp/make_fbank/${x}.${lang} ${fbankdir}/${x}.${lang}
        done

        # speed-perturbed
        utils/perturb_data_dir_speed.sh 0.9 ${data_dir}/train.${src_lang}-${tgt_lang}.${lang} ${data_dir}/temp1.${lang}
        utils/perturb_data_dir_speed.sh 1.0 ${data_dir}/train.${src_lang}-${tgt_lang}.${lang} ${data_dir}/temp2.${lang}
        utils/perturb_data_dir_speed.sh 1.1 ${data_dir}/train.${src_lang}-${tgt_lang}.${lang} ${data_dir}/temp3.${lang}
        utils/combine_data.sh --extra-files utt2uniq ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang} \
            ${data_dir}/temp1.${lang} ${data_dir}/temp2.${lang} ${data_dir}/temp3.${lang}
        rm -r ${data_dir}/temp1.${lang} ${data_dir}/temp2.${lang} ${data_dir}/temp3.${lang}
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang} exp/make_fbank/train_sp.${src_lang}-${tgt_lang}.${lang} ${fbankdir}/train_sp.${src_lang}-${tgt_lang}.${lang}
        #for lang in ${src_lang} ${tgt_lang}; do

        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.tc >${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.tc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc >${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc.rm >${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc.rm
        
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.tc >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.tc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc.rm >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc.rm
        
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.tc >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.tc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}.${lang}/text.lc.rm >>${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/text.lc.rm
        #done
    done

    for x in train_sp.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang} test.${src_lang}-${tgt_lang}; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in ${tgt_lang} ${src_lang}; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 300 ${data_dir}/${x}.${lang} ${data_dir}/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f 1 -d " " ${data_dir}/${x}.${src_lang}.tmp/text > ${data_dir}/${x}.${tgt_lang}.tmp/reclist1
        cut -f 1 -d " " ${data_dir}/${x}.${tgt_lang}.tmp/text > ${data_dir}/${x}.${tgt_lang}.tmp/reclist2
        comm -12 ${data_dir}/${x}.${tgt_lang}.tmp/reclist1 ${data_dir}/${x}.${tgt_lang}.tmp/reclist2 > ${data_dir}/${x}.${src_lang}.tmp/reclist

        for lang in ${tgt_lang} ${src_lang}; do
            reduce_data_dir.sh ${data_dir}/${x}.${lang}.tmp ${data_dir}/${x}.${src_lang}.tmp/reclist ${data_dir}/${x}.${lang}
            utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" ${data_dir}/${x}.${lang}
        done
        rm -rf ${data_dir}/${x}.*.tmp
    done
fi
#Assuming all the data preparation is done with ST.

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for lang in ${src_lang} ${tgt_lang}; do

        # compute global CMVN
        compute-cmvn-stats scp:${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/cmvn.ark

        dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
            ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/cmvn.ark exp/dump_feats/train_sp.${src_lang}-${tgt_lang}.${lang} ${dumpdir}/train_sp.${src_lang}-${tgt_lang}.${lang}/delta${do_delta}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${data_dir}/dev.${src_lang}-${tgt_lang}.${lang}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/cmvn.ark exp/dump_feats/dev.${src_lang}-${tgt_lang}.${lang} ${dumpdir}/dev.${src_lang}-${tgt_lang}.${lang}/delta${do_delta}
        #for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/test.${src_lang}-${tgt_lang}.${lang}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${data_dir}/test.${src_lang}-${tgt_lang}.${lang}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${lang}/cmvn.ark exp/dump_feats/recog/test.${src_lang}-${tgt_lang}.${lang} \
            ${feat_recog_dir}
        #done
    done
fi

#if [ ${use_st_dict} = true ]; then
if [ ${bpemode} != "word" ]; then
    dict_path=/home/soky/work1/data/trilingual
    dict=${dict_path}/lang_${bpemode}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_units_${case}.txt
    nlsyms=${dict_path}/lang_${bpemode}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}_non_lang_syms_${case}.txt
    bpemodel=${dict_path}/lang_${bpemode}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${case}
else
    dict=data/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_units_${case}.txt
    nlsyms=data/lang_${bpemode}/${train_set}_non_lang_syms_${case}.txt
    bpemodel=data/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_${case}
fi
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    
    #if [ ${use_st_dict} = false ]; then
    #mkdir -p data/lang_${bpemode}/

    # echo "make a non-linguistic symbol list for all languages"
    
    # grep sp1.0 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    # echo "<space> " >> ${nlsyms}
    # cat ${nlsyms}

    # echo "make a dictionary"
    # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    # offset=$(wc -l < ${dict})

    if [ ${bpemode} == "bpe" ]; then            
        # cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.${src_lang}/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_${bpemode}/input.txt
        # spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_${bpemode}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
        # spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_${bpemode}/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    
        # NOTE: ASR vocab is created with a source language only
        echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
        data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp,${dumpdir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_set}/text.${case} --bpecode ${bpemodel}.${src_lang}.model --lang ${src_lang} \
            ${data_dir}/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp,${dumpdir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_dev}/text.${case} --bpecode ${bpemodel}.${src_lang}.model --lang ${src_lang} \
            ${data_dir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp,${dumpdir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${rtask}/text.${case} --bpecode ${bpemodel}.${src_lang}.model --lang ${src_lang}  \
                ${data_dir}/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        done

        # update json (add source references)
        for x in train_sp.${src_lang}-${tgt_lang}.${tgt_lang} dev.${src_lang}-${tgt_lang}.${tgt_lang} test.${src_lang}-${tgt_lang}.${tgt_lang}; do
            xx=$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
            feat_dir=${dumpdir}/${xx}/delta${do_delta}
            datadir=${data_dir}/${x}
            update_json.sh --text ${datadir}/text.${case} --bpecode ${bpemodel}.${tgt_lang}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${data_dir}/${xx} ${dict}
        done

    elif [ ${bpemode} == "word" ]; then
        
        mkdir -p data/lang_${bpemode}/

        echo "make a non-linguistic symbol list for all languages"
        
        grep sp1.0 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
        cat ${nlsyms}

        echo "make a dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        offset=$(wc -l < ${dict})

        cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        
        # NOTE: ASR vocab is created with a source language only
        echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
        local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp,${dumpdir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_set}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${src_lang} \
            ${data_dir}/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
        local/data2json.sh --feat ${feat_dt_dir}/feats.scp,${dumpdir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_dev}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${src_lang} \
            ${data_dir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp,${dumpdir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${rtask}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${src_lang}  \
                ${data_dir}/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        done

        # update json (add source references)
        for x in train_sp.${src_lang}-${tgt_lang}.${tgt_lang} dev.${src_lang}-${tgt_lang}.${tgt_lang} test.${src_lang}-${tgt_lang}.${tgt_lang}; do
            xx=$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
            feat_dir=${dumpdir}/${xx}/delta${do_delta}
            datadir=${data_dir}/${x}
            local/update_json.sh --text ${datadir}/text.${case} --trans_type phn --nlsyms ${nlsyms} \
                ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${data_dir}/${xx} ${dict}
        done

    elif [ ${bpemode} == "character" ]; then

        # text2token.py -s 1 -n 1 <(cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case}) -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
        #     | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}

        # NOTE: ASR vocab is created with a source language only
        echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
        data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp,${dumpdir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_set}/text.${case} --nlsyms ${nlsyms} --lang ${src_lang} \
            ${data_dir}/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp,${dumpdir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${train_dev}/text.${case} --nlsyms ${nlsyms} --lang ${src_lang} \
            ${data_dir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp,${dumpdir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/delta${do_delta}/feats.scp --text ${data_dir}/${rtask}/text.${case} --nlsyms ${nlsyms} --lang ${src_lang}  \
                ${data_dir}/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        done

        # update json (add source references)
        for x in train_sp.${src_lang}-${tgt_lang}.${tgt_lang} dev.${src_lang}-${tgt_lang}.${tgt_lang} test.${src_lang}-${tgt_lang}.${tgt_lang}; do
            xx=$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
            feat_dir=${dumpdir}/${xx}/delta${do_delta}
            datadir=${data_dir}/${x}
            update_json.sh --text ${datadir}/text.${case} --nlsyms ${nlsyms} \
                ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${data_dir}/${xx} ${dict}
        done
    fi
    #fi
    

fi


# You can skip this and remove --rnnlm option in the recognition (stage 3)
# if [ -z ${lmtag} ]; then
#     lmtag=$(basename ${lm_config%.*})_${case}
# fi
# lmexpname=${train_set}_${case}_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
# lmexpdir=exp/${lmexpname}
# mkdir -p ${lmexpdir}

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: LM Preparation"
#     lmdatadir=${data_dir}/local/lm_${train_set}_${bpemode}${nbpe}
#     mkdir -p ${lmdatadir}
#     grep sp1.0 ${data_dir}/${train_set}/text.${case} | cut -f 2- -d " " | spm_encode --model=${bpemodel}.model --output_format=piece \
#         > ${lmdatadir}/train_${case}.txt
#     cut -f 2- -d " " ${data_dir}/${train_dev}/text.${case} | spm_encode --model=${bpemodel}.model --output_format=piece \
#         > ${lmdatadir}/valid_${case}.txt
#     ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#         lm_train.py \
#         --config ${lm_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --verbose 1 \
#         --outdir ${lmexpdir} \
#         --tensorboard-dir tensorboard/${lmexpname} \
#         --train-label ${lmdatadir}/train_${case}.txt \
#         --valid-label ${lmdatadir}/valid_${case}.txt \
#         --resume ${lm_resume} \
#         --dict ${dict}
# fi

if [ -z ${tag} ]; then
    expname=${train_set}_${case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json \
        --enc-init ${enc_init}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --join-train "st" \
            --model ${expdir}/results/${recog_model} #\--rnnlm ${lmexpdir}/rnnlm.model.best

        if [ ${bpemode} == "bpe" ]; then    
            local/score_sclite.sh --case ${case} --bpe ${nbpe} --bpemodel ${bpemodel}.${src_lang}.model --wer true \
                ${expdir}/${decode_dir} ${dict}
        else
            local/score_sclite.sh --case ${case} --nlsyms ${nlsyms} --wer true \
                ${expdir}/${decode_dir} ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
