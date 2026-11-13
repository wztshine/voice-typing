# -*- coding: utf-8 -*-
"""
热词支持
1) 解码期: 对热词对应的 token 序列加 log-prob 偏置(beam search 内部)
2) 输出期: 把识别结果里大小写/全半角不一致的英文热词替换成标准写法
"""
import os
import re
import traceback
from typing import List, Optional

import torch
import torch.nn.functional as F

_TRIE = None
_BONUS = 2.0
_TERMS = []


def _node():
    return {"ch": {}, "end": False}


def load_words(path):
    words = []
    if not os.path.exists(path):
        return words
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)
    return words


def build_trie(tokenizer, words, bonus=2.0):
    """把热词转成 token 序列的 Trie; 返回真正生效的词表"""
    global _TRIE, _BONUS, _TERMS
    root = _node()
    used = []
    skipped = []
    for w in words:
        try:
            _tokens, ids = tokenizer.tokenize(w)
        except Exception:
            skipped.append(w)
            continue
        # 0=<blank> 1=<unk> 2=<pad> 3=<sos> 4=<eos>
        if not ids or any(i <= 4 for i in ids):
            skipped.append(w)
            continue
        node = root
        for i in ids:
            node = node["ch"].setdefault(int(i), _node())
        node["end"] = True
        used.append(w)
    _TRIE = root if used else None
    _BONUS = float(bonus)
    _TERMS = used
    return used, skipped


def _step(state, tok):
    if state is not None:
        nxt = state["ch"].get(tok)
        if nxt is not None:
            return nxt
    if _TRIE is not None:
        return _TRIE["ch"].get(tok)
    return None


def _bias(t_scores, states):
    """给每个 beam 的下一步候选加偏置"""
    if _TRIE is None or _BONUS <= 0:
        return t_scores
    rows, vocab = t_scores.shape
    add = torch.zeros_like(t_scores)
    touched = False
    for i, st in enumerate(states):
        if st is None:
            continue
        for cid in st["ch"]:
            if cid < vocab:
                add[i, cid] = _BONUS
                touched = True
    if not touched:
        return t_scores
    return t_scores + add


def fix_text(text):
    """输出端拼写校正: 英文热词按标准大小写; 中文热词原样"""
    if not text or not _TERMS:
        return text
    for term in sorted(_TERMS, key=len, reverse=True):
        if term.isascii():
            pat = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
            text = re.sub(pat, term, text, flags=re.IGNORECASE)
        else:
            text = text.replace(term.lower(), term)
    return text


def install(tokenizer, hotword_path, bonus=2.0, logger=print):
    """加载热词并给 TransformerDecoder.batch_beam_search 打补丁"""
    words = load_words(hotword_path)
    if not words:
        return
    try:
        used, skipped = build_trie(tokenizer, words, bonus)
        if skipped:
            logger(f"热词跳过(含词典外字符/数字): {skipped}")
        if not used:
            logger("热词无可生效条目")
            return
        from fireredasr2s.fireredasr2.models.module.transformer_decoder import TransformerDecoder
        TransformerDecoder.batch_beam_search = _batch_beam_search
        logger(f"热词已启用 {len(used)} 条 (偏置 {bonus}): {used[:10]}")
    except Exception:
        logger("热词安装失败, 忽略:\n" + traceback.format_exc())


# 以下为原版 batch_beam_search 的复刻, 只增加 states/偏置两处逻辑
def _batch_beam_search(self, encoder_outputs, src_masks,
                       beam_size=1, nbest=1, decode_max_len=0,
                       softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                       elm=None, elm_weight=0.0):
    B = beam_size
    N, Ti, H = encoder_outputs.size()
    device = encoder_outputs.device
    maxlen = decode_max_len if decode_max_len > 0 else Ti
    assert eos_penalty > 0.0

    encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N * B, Ti, H)
    src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N * B, -1, Ti)
    ys = torch.ones(N * B, 1).fill_(self.sos_id).long().to(device)
    t_ys = ys.clone()
    confidences = torch.zeros(N * B, 1).float().to(device)
    caches: List[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]
    scores = torch.tensor([0.0] + [-self.INF] * (B - 1)).float().to(device)
    scores = scores.repeat(N).view(N * B, 1)
    is_finished = torch.zeros_like(scores)
    if elm is not None:
        elm_cache = elm.init_hidden(encoder_outputs, N * B)

    states = [_TRIE] * (N * B)

    for t in range(maxlen):
        tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

        dec_output = self.dropout(
            self.tgt_word_emb(ys) * self.scale +
            self.positional_encoding(ys))
        i = 0
        for dec_layer in self.layer_stack:
            dec_output = dec_layer.forward(
                dec_output, encoder_outputs,
                tgt_mask, src_mask,
                cache=caches[i])
            caches[i] = dec_output
            i += 1

        dec_output = self.layer_norm_out(dec_output)

        t_logit = self.tgt_word_prj(dec_output[:, -1])
        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

        if elm is not None and elm_weight > 0.0:
            elm_logit, elm_cache = elm.forward_model(t_ys, hidden=elm_cache)
            t_lm_scores = torch.log_softmax(elm_logit[:, -1], dim=-1) * (1 - is_finished.float())
            t_lm_scores[:, elm.eos_id] *= 3
            t_scores = t_scores + elm_weight * t_lm_scores

        t_scores = _bias(t_scores, states)

        if eos_penalty != 1.0:
            t_scores[:, self.eos_id] *= eos_penalty

        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
        t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
        t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

        scores = scores + t_topB_scores

        scores = scores.view(N, B * B)
        scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
        scores = scores.view(-1, 1)

        topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N * B)
        stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N * B).to(device)
        topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

        ys = ys[topB_row_number_in_ys]
        t_ys = torch.gather(t_topB_ys.view(N, B * B), dim=1, index=topB_score_ids).view(N * B, 1)
        ys = torch.cat((ys, t_ys), dim=1)

        # 更新热词状态(跟随剪枝后的 beam)
        if _TRIE is not None:
            new_states = []
            for row in range(N * B):
                prev = states[int(topB_row_number_in_ys[row].item())]
                new_states.append(_step(prev, int(t_ys[row, 0].item())))
            states = new_states

        confidences = confidences[topB_row_number_in_ys]
        t_confidences = torch.gather(t_topB_scores.view(N, B * B), dim=1, index=topB_score_ids).view(N * B, 1)
        t_confidences = torch.exp(t_confidences)
        confidences = torch.cat((confidences, t_confidences), dim=1)

        new_caches: List[Optional[torch.Tensor]] = []
        for cache in caches:
            if cache is not None:
                new_caches.append(cache[topB_row_number_in_ys])
        caches = new_caches
        if elm and elm_weight > 0.0:
            elm_cache = (elm_cache[0][:, topB_row_number_in_ys], elm_cache[1][:, topB_row_number_in_ys])

        is_finished = t_ys.eq(self.eos_id)
        if is_finished.sum().item() == N * B:
            break

    scores = scores.view(N, B)
    ys = ys.view(N, B, -1)
    ys_lengths = self.get_ys_lengths(ys)
    if length_penalty > 0.0:
        penalty = torch.pow((5 + ys_lengths.float()) / (5.0 + 1), length_penalty)
        scores /= penalty
    nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
    nbest_scores = -1.0 * nbest_scores
    index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
    nbest_ys = ys.view(N * B, -1)[index.view(-1)]
    nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
    nbest_ys_lengths = ys_lengths.view(N * B)[index.view(-1)].view(N, -1)
    nbest_confidences = confidences.view(N * B, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)

    nbest_hyps = []
    for n in range(N):
        n_nbest_hyps = []
        for i, score in enumerate(nbest_scores[n]):
            confidence = nbest_confidences[n, i, 1:nbest_ys_lengths[n, i]]
            confidence = confidence.mean()
            n_nbest_hyps.append({
                "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]],
                "confidence": confidence,
            })
        nbest_hyps.append(n_nbest_hyps)
    return nbest_hyps
