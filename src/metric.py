import collections
import json
import math

from tqdm import tqdm

from transformers import BertTokenizer


class Metric:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.model_name_or_path = args.model_name_or_path

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)

    def get_best_indexes(self, logits, n_best_size):
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def generate_predictions(self, all_examples, all_features, all_results, n_best_size, max_answer_length):
        example_index_to_feature = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_feature[feature["example_index"]].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple("PrelimPrediction", ["feature_index", "start_index",
                                                                        "end_index", "start_logit", "end_logit"])
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        for (example_index, example) in enumerate(tqdm(all_examples)):
            features = example_index_to_feature[example_index]
            prelim_predictions = []

            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature["unique_id"]]
                start_indexes = self.get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self.get_best_indexes(result.end_logits, n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(feature["tokens"]):
                            continue
                        if end_index >= len(feature["tokens"]):
                            continue
                        if start_index not in feature["token_to_orig_map"]:
                            continue
                        if end_index not in feature["token_to_orig_map"]:
                            continue
                        if not feature["token_is_max_context"].get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

            seen_predictions = {}
            nbest = []

            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature["tokens"][pred.start_index:(pred.end_index + 1)]
                    final_text = self.tokenizer.convert_tokens_to_string(tok_tokens)
                    final_text = final_text.replace(' ', '')
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        start_index=pred.start_index,
                        end_index=pred.end_index))

            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self.compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            all_predictions[example["qid"]] = nbest_json[0]["text"]
            all_nbest_json[example["qid"]] = nbest_json

        return all_predictions

    def remove_punctuation(self, in_str):
        in_str = str(in_str).lower().strip()
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)

    def find_lcs(self, s1, s2):
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax

    def calc_f1_score(self, answers, prediction):
        f1_scores = []
        for ans in answers:
            ans_segs = self.tokenizer.tokenize(ans)
            prediction_segs = self.tokenizer.tokenize(prediction)
            lcs, lcs_len = self.find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * lcs_len / len(prediction_segs)
            recall = 1.0 * lcs_len / len(ans_segs)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)

    def calc_em_score(self, answers, prediction):
        em = 0
        for ans in answers:
            ans_ = self.remove_punctuation(ans)
            prediction_ = self.remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em

    def evaluate(self, ground_truth_file, predictions):
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0
        for instance in ground_truth_file["data"]:
            for para in instance["paragraphs"]:
                for qas in para['qas']:
                    total_count += 1
                    query_id = qas['id'].strip()
                    query_text = qas['question'].strip()
                    answers = [x["text"] for x in qas['answers']]

                    if query_id not in predictions.keys():
                        skip_count += 1
                        continue

                    prediction = str(predictions[query_id])
                    f1 += self.calc_f1_score(answers, prediction)
                    em += self.calc_em_score(answers, prediction)
        print('total_count', total_count)
        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count
        return f1_score, em_score, total_count, skip_count

    def get_eval(self, original_file, predictions):
        ground_truth_file = json.load(open(original_file, 'r', encoding='utf-8'))
        F1, EM, TOTAL, SKIP = self.evaluate(ground_truth_file, predictions)
        AVG = (EM + F1) * 0.5
        output_result = collections.OrderedDict()
        output_result['AVERAGE'] = '%.3f' % AVG
        output_result['F1'] = '%.3f' % F1
        output_result['EM'] = '%.3f' % EM
        output_result['TOTAL'] = TOTAL
        output_result['SKIP'] = SKIP
        print('output_result', output_result)
        return output_result
