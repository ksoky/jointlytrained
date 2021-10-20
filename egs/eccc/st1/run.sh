#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# pre-training related
asr_model=
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
eccc=/home/soky/ECCC/wav/ST #raw wav and text
data_dir=/home/soky/work1/data/trilingual #clean fbank, text
dumpdir=${data_dir}/dump/st    # directory to dump full features
# target language related
src_lang=en
tgt_lang=km
# you can choose from de, es, fr, it, nl, pt, ro, ru
# if you want to train the multilingual model, segment languages with _ as follows:
# e.g., tgt_lang="de_es_fr"
# if you want to use all languages, set tgt_lang="all"

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

train_set=train_sp.${src_lang}-${tgt_lang}.${tgt_lang}
train_dev=dev.${src_lang}-${tgt_lang}.${tgt_lang}
trans_set=""
#dev.${slang}-${tlang}.${tlang} 
for slang in $(echo ${src_lang} | tr '_' ' '); do  
    for tlang in $(echo ${tgt_lang} | tr '_' ' '); do
        if [ ${slang} != ${tlang} ]; then
            trans_set="${trans_set} test.${slang}-${tlang}.${tlang}"
        fi
    done
done

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage 0: Data Preparation"
    for slang in $(echo ${src_lang} | tr '_' ' '); do
        for tlang in $(echo ${tgt_lang} | tr '_' ' '); do
            if [ ${slang} != ${tlang} ]; then
                local/data_prep_eccc.sh ${eccc} ${data_dir} ${slang} ${tlang}
            fi
        done
    done
fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data Preparation"
#     for lang in $(echo ${tgt_lang} | tr '_' ' '); do
#         local/data_prep.sh ${must_c} ${lang}
#     done
# fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=${data_dir}/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for tlang in $(echo ${tgt_lang} | tr '_' ' '); do
        for slang in $(echo ${src_lang} | tr '_' ' '); do
            if [ ${slang} != ${tlang} ]; then
                for x in train.${slang}-${tlang} dev.${slang}-${tlang} test.${slang}-${tlang}; do
                    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
                        ${data_dir}/${x} exp/make_fbank/${x} ${fbankdir}/${x}
                done
            fi
        done
    done

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 ${data_dir}/train.${src_lang}-${tgt_lang} ${data_dir}/temp1.${tgt_lang}
    utils/perturb_data_dir_speed.sh 1.0 ${data_dir}/train.${src_lang}-${tgt_lang} ${data_dir}/temp2.${tgt_lang}
    utils/perturb_data_dir_speed.sh 1.1 ${data_dir}/train.${src_lang}-${tgt_lang} ${data_dir}/temp3.${tgt_lang}
    utils/combine_data.sh --extra-files utt2uniq ${data_dir}/train_sp.${src_lang}-${tgt_lang} \
        ${data_dir}/temp1.${tgt_lang} ${data_dir}/temp2.${tgt_lang} ${data_dir}/temp3.${tgt_lang}
    rm -r ${data_dir}/temp1.${tgt_lang} ${data_dir}/temp2.${tgt_lang} ${data_dir}/temp3.${tgt_lang}
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        ${data_dir}/train_sp.${src_lang}-${tgt_lang} exp/make_fbank/train_sp.${src_lang}-${tgt_lang} ${fbankdir}
    for lang in ${src_lang} ${tgt_lang}; do
        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.tc.${lang} >${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.${lang} >${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.rm.${lang} >${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.rm.${lang}
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.tc.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.rm.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.rm.${lang}
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/train.${src_lang}-${tgt_lang}/utt2spk > ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.tc.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 ${data_dir}/train_sp.${src_lang}-${tgt_lang}/utt_map <${data_dir}/train.${src_lang}-${tgt_lang}/text.lc.rm.${lang} >>${data_dir}/train_sp.${src_lang}-${tgt_lang}/text.lc.rm.${lang}
    done

    # Divide into source and target languages
    for x in train_sp.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang} test.${src_lang}-${tgt_lang}; do
        local/divide_lang_eccc.sh ${x} ${src_lang} ${tgt_lang} ${data_dir}
    done

    for x in train_sp.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang}; do
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

    # compute global CMVN
    compute-cmvn-stats scp:${data_dir}/${train_set}/feats.scp ${data_dir}/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        ${data_dir}/${train_set}/feats.scp ${data_dir}/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        ${data_dir}/${train_dev}/feats.scp ${data_dir}/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for ttask in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${data_dir}/${ttask}/feats.scp ${data_dir}/${train_set}/cmvn.ark exp/dump_feats/trans/${ttask} \
            ${feat_trans_dir}
    done
fi

dict=${data_dir}/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=${data_dir}/lang_${bpemode}/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=${data_dir}/lang_${bpemode}/${train_set}_${bpemode}${nbpe}_${tgt_case}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${data_dir}/lang_${bpemode}/

    echo "make a non-linguistic symbol list for all languages"
    
    grep sp1.0 ${data_dir}/train_sp.${src_lang}-${tgt_lang}.*/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    

    if [ ${bpemode} == "bpe" ]; then
        for lang in ${tgt_lang} ${src_lang}; do

            offset=$(wc -l < ${dict})

            cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' | sort -u > ${data_dir}/lang_${bpemode}/input.txt
            spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=${data_dir}/lang_${bpemode}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel}.${lang} --input_sentence_size=100000000 --character_coverage=1.0
            spm_encode --model=${bpemodel}.${lang}.model --output_format=piece < ${data_dir}/lang_${bpemode}/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        done
        wc -l ${dict}

        echo "make json files"
        data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
            ${data_dir}/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/${train_dev}/text.${tgt_case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
            ${data_dir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        for ttask in ${trans_set}; do
            feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
            data2json.sh --feat ${feat_trans_dir}/feats.scp --text ${data_dir}/${ttask}/text.${tgt_case} --bpecode ${bpemodel}.${tgt_lang}.model --lang ${tgt_lang} \
                ${data_dir}/${ttask} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev}; do
            feat_dir=${dumpdir}/${x}/delta${do_delta}
            datadir=${data_dir}/$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
            update_json.sh --text ${datadir}/text.${src_case} --bpecode ${bpemodel}.${src_lang}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${datadir} ${dict}
        done

    elif [ ${bpemode} == "word" ]; then
        
        cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        
        # NOTE: ASR vocab is created with a source language only
        echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
        local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} \
            ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
        local/data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} \
            ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --text ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --trans_type phn --nlsyms ${nlsyms} \
                ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${recog_set}; do
        
            feat_dir=${dumpdir}/${x}/delta${do_delta}
            datadir=${data_dir}/${x}
            local/update_json.sh --text ${datadir}/text.${case} --trans_type phn \
                ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${datadir} ${dict}
                
        done

    elif [ ${bpemode} == "character" ]; then

        text2token.py -s 1 -n 1 <(cat ${data_dir}/{train_sp,dev,test}.${src_lang}-${tgt_lang}.*/text.${case}) -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
            | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}

        # NOTE: ASR vocab is created with a source language only
        echo "make json files with $(wc -l < ${dict}) ${bpemode}s"
        data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} \
            ${data_dir}/train_sp.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} \
            ${data_dir}/dev.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp --text ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}/text.${case} --nlsyms ${nlsyms} \
                ${data_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${recog_set}; do
        
            feat_dir=${dumpdir}/${x}/delta${do_delta}
            datadir=${data_dir}/${x}
            update_json.sh --text ${datadir}/text.${case} --nlsyms ${nlsyms} \
                ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${datadir} ${dict}
        done
    fi
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
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
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi

    pids=() # initialize pids
    for ttask in ${trans_set}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        score_bleu.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.${tgt_lang}.model \
            --remove_nonverbal ${remove_nonverbal} \
            ${expdir}/${decode_dir} ${tgt_lang} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
