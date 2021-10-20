# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Join ASR and MT model (pytorch)."""

from argparse import Namespace
from itertools import groupby
import logging
import math

import chainer
from chainer import reporter
import numpy
import torch

from espnet.utils.cli_utils import strtobool

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
#from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask #for MT
#from espnet.nets.pytorch_backend.nets_utils import pad_list #for ST
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.add_sos_eos import mask_uniform
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder_joint import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder_dual_st import DualEncoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args


CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """Define a chainer reporter wrapper."""
    def report(self, loss_ctc, loss_att, cer_ctc, cer, wer, acc, ppl, bleu, mtl_loss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"ppl": ppl}, self)
        reporter.report({"bleu": bleu}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")
        group = add_arguments_transformer_common(group)
        # Non-autoregressive training
        group.add_argument(
            "--decoder-mode",
            default="AR",
            type=str,
            choices=["ar", "maskctc"],
            help="AR: standard autoregressive training, "
            "maskctc: non-autoregressive training based on Mask CTC",
        )
        group.add_argument(
            "--report-bleu",
            default=True,
            action="store_true",
            help="Compute BLEU on development set",
        )
        # multilingual related
        group.add_argument(
            "--multilingual",
            default=False,
            type=strtobool,
            help="Prepend target language ID to the source sentence. "
            "Both source/target language IDs must be prepend in the pre-processing stage.",
        )
        group.add_argument(
            "--replace-sos",
            default=False,
            type=strtobool,
            help="Replace <sos> in the decoder with a target language ID "
            "(the first token in the target sequence)",
        )
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        ##Join ASR and MT/ST encoder
        self.encoder = DualEncoder(
            asr_idim=idim,
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer="conv2d",
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            self.criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None
        self.blank = 0
        self.pad = 0  # use <blank> for padding
        self.decoder_mode = args.decoder_mode
        if self.decoder_mode == "maskctc":
            self.mask_token = odim - 1
            self.sos = odim - 2
            self.eos = odim - 2
        else:
            self.sos = odim - 1
            self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        
        

        self.reporter = Reporter()
        # submodule for ASR task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = args.asr_weight
        if self.asr_weight > 0 and args.mtlalpha < 1:
            self.decoder_asr = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )

        # submodule for MT task
        self.mt_weight = args.mt_weight
        if self.mt_weight > 0:
            self.encoder_mt = Encoder(
                idim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer="embed",
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                padding_idx=0,
            )

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator_asr = ASRErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator_asr = None

        self.normalize_length = args.transformer_length_normalized_loss  # for PPL
        self.error_calculator_mt = MTErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank, args.report_bleu
        )
        
        # multilingual related
        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)

        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            torch.nn.init.normal_(
                self.encoder_mt.embed[0].weight, mean=0, std=args.adim ** -0.5
            )
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)

    #xs_pad, ilens, ys_pad, ys_pad_src, src_lens
    def forward(self, xs_pad, ilens, xt_pad, itlens, ys_pad, ys_pad_src):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor xt_pad: batch of padded source-st sequence (B, idim)
        :param torch.Tensor itlens: batch of lengths of source-st sequence (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        tgt_lang_ids = None
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder
        # xs_pad is the source of ASR (fbank)
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # xt_pad is the source of MT
        xt_pad = xt_pad[:, : max(itlens)]  # for data parallel
        xt_mask = make_non_pad_mask(itlens.tolist()).to(xt_pad.device).unsqueeze(-2)


        
        #from pudb import set_trace; set_trace()
        hs_pad, hs_mask, st_pad, st_mask = self.encoder(xs_pad, src_mask, xt_pad, xt_mask)

        self.hs_pad = hs_pad
        
        # 2. forward decoder
        if self.decoder is not None:
            if self.decoder_mode == "maskctc":
                ys_in_pad, ys_out_pad = mask_uniform(
                    ys_pad, self.mask_token, self.eos, self.ignore_id
                )
                ys_mask = (ys_in_pad != self.ignore_id).unsqueeze(-2)
            else:
                ys_in_pad, ys_out_pad = add_sos_eos(
                    ys_pad, self.sos, self.eos, self.ignore_id
                )
                ys_mask = target_mask(ys_in_pad, self.ignore_id)
            # replace <sos> with target language ID
            if self.replace_sos:
                ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
                ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask, st_pad, st_mask)
            self.pred_pad = pred_pad
            
            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator_asr is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator_asr(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator_asr is None or self.decoder is None:
            cer, wer, self.bleu = None, None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator_asr(ys_hat.cpu(), ys_pad.cpu())
            self.bleu = self.error_calculator_mt(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)

        if self.normalize_length:
            self.ppl = numpy.exp(loss_data)
        else:
            batch_size = ys_out_pad.size(0)
            ys_out_pad = ys_out_pad.view(-1)
            ignore = ys_out_pad == self.ignore_id  # (B*T,)
            total_n_tokens = len(ys_out_pad) - ignore.sum().item()
            self.ppl = numpy.exp(loss_data * batch_size / total_n_tokens)
       
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, cer_ctc, cer, wer, self.acc, self.ppl, self.bleu, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    # def encode(self, x):
    #     """Encode acoustic features.

    #     :param ndarray x: source acoustic feature (T, D)
    #     :return: encoder outputs
    #     :rtype: torch.Tensor
    #     """
    #     self.eval()
    #     x = torch.as_tensor(x).unsqueeze(0)
    #     enc_output, _ = self.encoder(x, None)
    #     return enc_output.squeeze(0)
    def encode(self, x, xt):
        self.eval()
        
        x = torch.as_tensor(x).unsqueeze(0)
        xt = torch.as_tensor(xt)
        x_enc_output, _, xt_enc_output, _ = self.encoder(x, None, xt, None)
        
        return x_enc_output.squeeze(0), xt_enc_output.squeeze(0)

    def recognize(self, x, xt, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(recog_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(recog_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        #logging.info("input lengths: " + str(xt.size(0)))
        #xt_pad = xt.unsqueeze(0)
        #tgt_lang = None
        #if recog_args.tgt_lang:recog_args.tgt_lang
        #tgt_lang = char_list.index("en")
        #xt_pad, _ = self.target_forcing(xt_pad, tgt_lang=tgt_lang)

        asr_output, st_output = self.encode(x, xt)
        #logging.info("Is it the probem here!")
        asr_output, st_output = asr_output.unsqueeze(0), st_output.unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(asr_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(asr_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = asr_output.squeeze(0)
        ht = st_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []
        
        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)

                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, asr_output, st_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, asr_output, st_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, asr_output, st_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, xt_pad, itlens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, xt_pad, itlens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, xt_pad, itlens, ys_pad, ys_pad_src):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, xt_pad, itlens, ys_pad, ys_pad_src)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret


    def target_forcing(self, xs_pad, ys_pad=None, tgt_lang=None):
        """Prepend target language IDs to source sentences for multilingual MT.

        These tags are prepended in source/target sentences as pre-processing.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :return: source text without language IDs
        :rtype: torch.Tensor
        :return: target text without language IDs
        :rtype: torch.Tensor
        :return: target language IDs
        :rtype: torch.Tensor (B, 1)
        """
        if self.multilingual:
            xs_pad = xs_pad[:, 1:]  # remove source language IDs here
            if ys_pad is not None:
                # remove language ID in the beginning
                lang_ids = ys_pad[:, 0].unsqueeze(1)
                ys_pad = ys_pad[:, 1:]
            elif tgt_lang is not None:
                lang_ids = xs_pad.new_zeros(xs_pad.size(0), 1).fill_(tgt_lang)
            else:
                raise ValueError("Set ys_pad or tgt_lang.")

            # prepend target language ID to source sentences
            xs_pad = torch.cat([lang_ids, xs_pad], dim=1)
        return xs_pad, ys_pad
