#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=4            # number of parallel jobs for decoding
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config= #conf/specaug.yaml
train_config=conf/train_join.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# preprocessing related
case=lc.rm
#tgt_case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
sfisher_speech=~/work1/data/fisher_callhome_spanish/links/LDC2010S01
sfisher_transcripts=~/work1/data/fisher_callhome_spanish/links/LDC2010T04
split=local/splits/split_fisher

callhome_speech=~/work1/data/fisher_callhome_spanish/links/LDC96S35
callhome_transcripts=~/work1/data/fisher_callhome_spanish/links/LDC96T17
split_callhome=local/splits/split_callhome

# bpemode (unigram or bpe)
nbpe=1000
bpemode=bpe
# NOTE: nbpe=53 means character-level ST (lc.rm)
# NOTE: nbpe=66 means character-level ST (lc)
# NOTE: nbpe=98 means character-level ST (tc)

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train.es
train_set_prefix=train
train_dev=train_dev.es
recog_set="fisher_dev.es fisher_dev2.es fisher_test.es callhome_devtest.es callhome_evltest.es"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/fsp_data_prep.sh ${sfisher_speech} ${sfisher_transcripts}
    local/callhome_data_prep.sh ${callhome_speech} ${callhome_transcripts}

    # split data
    local/create_splits.sh ${split}
    local/callhome_create_splits.sh ${split_callhome}

    # concatenate multiple utterances
    local/normalize_trans.sh ${sfisher_transcripts} ${callhome_transcripts}
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        ../asr1b/data/${train_set}/feats.scp ../asr1b/data/train_sp.es/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    # for x in ${train_dev} ${recog_set}; do
    #     feat_trans_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_trans_dir}
    #     dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    #         ../asr1b/data/${x}/feats.scp ../asr1b/data/${train_set}/cmvn.ark exp/dump_feats/recog/${x} \
    #         ${feat_trans_dir}
    # done
fi

dict=data/lang_1spm/train_sp.es_${bpemode}${nbpe}_units_${case}.txt
nlsyms=data/lang_1spm/train_sp.es_non_lang_syms_${case}.txt
bpemodel=data/lang_1spm/train_sp.es_${bpemode}${nbpe}_${case}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    # mkdir -p data/lang_1spm/

    # echo "make a non-linguistic symbol list for all languages"
    # grep sp1.0 ../asr1b/data/${train_set_prefix}.*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    # cat ${nlsyms}

    # echo "make a joint source and target dictionary"
    # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    # for lang in es en; do
    #     grep sp1.0 ../asr1b/data/${train_set_prefix}.${lang}/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_1spm/input.txt
    #     spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel}.${lang} --input_sentence_size=100000000 --character_coverage=1.0
    #     spm_encode --model=${bpemodel}.${lang}.model --output_format=piece < data/lang_1spm/input.txt >> data/lang_1spm/all_input.txt
    # done

    # offset=$(wc -l < ${dict})
    # cat data/lang_1spm/all_input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    
    wc -l ${dict}

    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ../asr1b/data/${train_set}/text.${case} --bpecode ${bpemodel}.es.model --lang es \
        ../asr1b/data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
    # for x in ${train_dev} ${recog_set}; do
    #     feat_trans_dir=${dumpdir}/${x}/delta${do_delta}
    #     data2json.sh --feat ${feat_trans_dir}/feats.scp --text ../asr1b/data/${x}/text.${case} --bpecode ${bpemodel}.es.model --lang es \
    #         ../asr1b/data/${x} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${case}.json
    # done

    # # update json (add source references)
    # for x in ${train_set} ${train_dev} ${recog_set}; do
    #     feat_dir=${dumpdir}/${x}/delta${do_delta}
    #     data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${tgt_lang}.${tgt_lang}
    #     update_json.sh --text ${data_dir}/text.${tgt_case} --bpecode ${bpemodel}.${tgt_lang}.model \
    #         ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
    # done


    # Fisher has 4 references per utterance
    # for x in fisher_dev.en fisher_dev2.en fisher_test.en; do
    #     feat_trans_dir=${dumpdir}/${x}/delta${do_delta}
    #     for no in 1 2 3; do
    #         data2json.sh --text data/${x}/text.${tgt_case}.${no} --feat ${feat_trans_dir}/feats.scp --bpecode ${bpemodel}.model --lang en \
    #             data/${x} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}_${no}.${tgt_case}.json
    #     done
    # done

    # update json (add source references)${train_dev} ${recog_set}
    for x in ${train_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=../asr1b/data/$(echo ${x} | cut -f 1 -d ".").en
        update_json.sh --text ${data_dir}/text.${case} --bpecode ${bpemodel}.en.model \
            ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${data_dir} ${dict}
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 3)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})_${case}
fi
lmexpname=train_sp.es_${case}_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

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
            --join-train "mt" \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        if [ ${bpemode} == "bpe" ]; then    
            score_sclite_case.sh --case ${case} --bpe ${nbpe} --bpemodel ${bpemodel}.es.model --wer true \
                ${expdir}/${decode_dir} ${dict}
        else
            score_sclite_case.sh --case ${case} --nlsyms ${nlsyms} --wer true \
                ${expdir}/${decode_dir} ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
