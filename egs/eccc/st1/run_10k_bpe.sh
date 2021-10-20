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
data_dir=/home/soky/work1/data/trilingual #clean fbank, text
dumpdir=dumpmltunits    # directory to dump full features
# target language related
src_lang=en
tgt_lang=km

# you can choose from de, es, fr, it, nl, pt, ro, ru
# if you want to train the multilingual model, segment languages with _ as follows:
# e.g., tgt_lang="de_es_fr"
# if you want to use all languages, set tgt_lang="all"

# use the same dict as in the ST task
use_st_dict=false

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

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data Download"
#     for lang in $(echo ${tgt_lang} | tr '_' ' '); do
#         local/download_and_untar.sh ${must_c} ${lang}
#     done
# fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data Preparation"
#     for lang in $(echo ${tgt_lang} | tr '_' ' '); do
#         local/data_prep.sh ${must_c} ${lang}
#     done
# fi

#Assuming all the data preparation is done with ST.

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    
    # dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
    #     ${data_dir}/${train_set}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    # dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    #     ${data_dir}/${train_dev}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${data_dir}/${rtask}/feats.scp ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

if [ ${bpemode} == "bpe" ]; then
    dict=data/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_units_${case}.txt
    nlsyms=data/lang_${bpemode}/${train_set}_non_lang_syms_${case}.txt
    bpemodel=${data_dir}/lang_${bpemode}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${case}
else
    dict=data/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_units_${case}.txt
    nlsyms=data/lang_${bpemode}/${train_set}_non_lang_syms_${case}.txt
    bpemodel=data/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_${case}
fi
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    
    if [ ${use_st_dict} = false ]; then
        mkdir -p data/lang_${bpemode}/

        echo "make a non-linguistic symbol list for all languages"
        
        grep sp1.0 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
        #echo "<space> " >> ${nlsyms}
        cat ${nlsyms}

        # echo "make a dictionary"
        # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        # offset=$(wc -l < ${dict})

        if [ ${bpemode} == "bpe" ]; then      

            # for lang in ${tgt_lang} ${src_lang}; do

            #     offset=$(wc -l < ${dict})

            #     cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.${lang}/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' | sort -u > ${data_dir}/lang_${bpemode}/input.txt
            #     #spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=${data_dir}/lang_${bpemode}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel}.${lang} --input_sentence_size=100000000 --character_coverage=1.0
            #     spm_encode --model=${bpemodel}.${lang}.model --output_format=piece < ${data_dir}/lang_${bpemode}/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
            # done

            wc -l ${dict}
            # cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.${src_lang}/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_${bpemode}/input.txt
            # spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_${bpemode}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
            # spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_${bpemode}/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        
            # NOTE: ASR vocab is created with a source language only
            echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
            # data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
            #     ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
            # data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
            #     ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
            for rtask in ${recog_set}; do
                feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
                data2json.sh --feat ${feat_recog_dir}/feats.scp --text ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
                    ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
            done

            # update json (add source references)${train_set} ${train_dev} 
            for x in ${recog_set}; do
                #feat_dir=${dumpdir}/${x}/delta${do_delta}
                # local/filtering_samples.py \
                #     --config ${train_config} \
                #     --data-json ${feat_dir}/data_${bpemode}${nbpe}.${case}.json \
                #     --mode-subsample "asr" \
                #     ${min_io_delta:+--min-io-delta $min_io_delta} \
                #     --output-json-path ${feat_dir}/data_${bpemode}${nbpe}.${case}.json

                feat_dir=${dumpdir}/${x}/delta${do_delta}
                datadir=${data_dir}/${x}
                update_json.sh --text ${datadir}/text.${case} --bpecode ${bpemodel}.${src_lang}.model \
                    ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${datadir} ${dict}
            done

        elif [ ${bpemode} == "word" ]; then
            
            #cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
            
            # NOTE: ASR vocab is created with a source language only
            echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
            # local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${tgt_lang} \
            #     ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
            # local/data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${tgt_lang} \
            #     ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
            for rtask in ${recog_set}; do
                feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
                local/data2json.sh --feat ${feat_recog_dir}/feats.scp --text ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} --lang ${tgt_lang} \
                    ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
            done

            # update json (add source references)${train_set} ${train_dev} 
            for x in ${recog_set}; do
            
                feat_dir=${dumpdir}/${x}/delta${do_delta}
                datadir=${data_dir}/${x}
                local/update_json.sh --text ${datadir}/text.${case} --trans_type phn \
                    ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${datadir} ${dict}
                    
            done

        elif [ ${bpemode} == "character" ]; then

            # text2token.py -s 1 -n 1 <(cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case}) -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
            #     | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}

            # NOTE: ASR vocab is created with a source language only
            echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
            # data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} --lang ${tgt_lang} \
            #     ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
            # data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} --lang ${tgt_lang} \
            #     ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
            for rtask in ${recog_set}; do
                feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
                data2json.sh --feat ${feat_recog_dir}/feats.scp --text ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} --lang ${tgt_lang} \
                    ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
            done

            # update json (add source references)${train_set} ${train_dev} 
            for x in ${recog_set}; do
            
                feat_dir=${dumpdir}/${x}/delta${do_delta}
                datadir=${data_dir}/${x}
                update_json.sh --text ${datadir}/text.${case} --nlsyms ${nlsyms} \
                    ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${datadir} ${dict}
            done
        fi
    fi
    

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
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
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
            --join-train true \
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
