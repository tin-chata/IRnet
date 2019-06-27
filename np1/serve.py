"""
Created on 2019-02-20
@author: duytinvo
"""
import os
import csv
import time
import torch
from np1.utils.gen_ans_from_fr import fr_to_ans
from np1.model import Translator_model
from np1.utils.data_utils import SaveloadHP, Csvfile


class NP1(object):
    def __init__(self, model_args="./data/trained_model/test_translator.args",
                 NL_Entities_filename='data/ontology/accounting_NL-Entities.csv',
                 nl_entity_groupablebyCSV_filename='data/ontology/accounting_nl_entity_groupableby.csv',
                 nl_entity_filterablebyCSV_filename='data/ontology/accounting_nl_entity_filterableby.csv',
                 RunningFolderPath='.',
                 use_cuda=False):
        margs = SaveloadHP.load(model_args)
        margs.use_cuda = use_cuda
        self.translator = Translator_model(margs)

        encoder_filename = os.path.join(margs.model_dir, margs.encoder_file)
        print("Load Model from file: %s" % encoder_filename)
        self.translator.encoder.load_state_dict(torch.load(encoder_filename))
        self.translator.encoder.to(self.translator.device)

        decoder_filename = os.path.join(margs.model_dir, margs.decoder_file)
        print("Load Model from file: %s" % decoder_filename)
        self.translator.decoder.load_state_dict(torch.load(decoder_filename))
        self.translator.decoder.to(self.translator.device)
        self.mechanical_reverse = fr_to_ans(
            RunningFolderPath=RunningFolderPath,
            NL_Entities_filename=NL_Entities_filename,
            nl_entity_groupablebyCSV_filename=nl_entity_groupablebyCSV_filename,
            nl_entity_filterablebyCSV_filename=nl_entity_filterablebyCSV_filename
        )

    def greedy_translate(self, nl):
        sql = self.translator.greedy_predict(nl)
        return [{"nl": nl, "sql": sql[0], "probability": sql[1]}]

    def beam_translate(self, nl, bw=2, topk=2):
        sqls = self.translator.beam_predict(nl, bw, topk)
        resp = {}
        resp['english_query'] = nl
        resp['parsed_queries'] = []
        resp['confidence_probabilities'] = []
        for sql in sqls:
            # resp.append({"nl": nl, "sql": sql[0], "probability": sql[1]})
            # Remove SOT and EOT
            resp['parsed_queries'].append(sql[0][4: -5])
            resp['confidence_probabilities'].append(sql[1])
        return resp

    def reverse_translate(self, resp):
        resp['answers'] = [""]
        try:
            answer = self.mechanical_reverse.translate(resp['parsed_queries'][0])
            if answer == self.mechanical_reverse.reverse_translation_error:
                raise Exception('reverse translation error')
            resp['answers'] = [answer]
        except:
            print('neural_parser encountered a problem during reverse inference.')

        return resp

    def regression_test(self, readfile, writefile, threshold=0.9, firstline=True):
        c = 0
        t = 0
        fail = 0
        low = 0
        print("\nREGRESSION TEST: ...")
        start = time.time()
        data = Csvfile(readfile, firstline=firstline)
        if len(data) > 0:
            with open(writefile, "w", newline='') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["english_query", "confident_score", "formal_representation", "error_type"])
                for source, target in data:
                    t += 1
                    sentence = " ".join(source)
                    decoded_batch = self.translator.beam_predict(sentence)
                    # Remove SOT and EOT
                    pred_target = decoded_batch[0][0][4:-5]
                    pred_prob = decoded_batch[0][1]
                    if pred_target != " ".join(target):
                        csvwriter.writerow([sentence, pred_prob, pred_target, "INCORRECT_FR"])
                        # print("\nINCORRECT FR (p = %.4f)" % pred_prob)
                        # print(sentence, '<-->', " ".join(target), '-->', pred_target)
                        c += 1
                        fail += 1
                    else:
                        if pred_prob < threshold:
                            csvwriter.writerow([sentence, pred_prob, pred_target, "LOW_SCORE"])
                            # print("\nLOW SCORE (p = %.4f)" % pred_prob)
                            # print(sentence, '<-->', " ".join(target), '-->', pred_target)
                            c += 1
                            low += 1
                        # if t % 1000 == 0:
                        #     print("\nINFO: - Processing %d queries in %.4f (mins)" % (t, (time.time() - start) / 60))
            end = (time.time() - start)/60
            print("\nREGRESSION: - Consumed time: %.2f (mins)" % end)
            print("            - Input file: %s" % readfile)
            print("            - Output file: %s" % writefile)
            print("            - Accuracy: %.4f (%%); Mislabelling: %.4f (%%)" % (100*(1-c/t), 100*(c/t)))
            print("            - Total failed queries: %d" % fail)
            print("            - Total low_conf queries: %d" % low)
        else:
            print("\nWARNING: The regression file is EMPTY")
        return


def main(nl="total sales from DUCKLING_TIME"):
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--model_args', help='Args file', default="./data/trained_model/test_translator.args",
                           type=str)

    argparser.add_argument('--NL_Entities_filename', help='Args file',
                           default='data/ontology/accounting_NL-Entities.csv',
                           type=str)

    argparser.add_argument('--nl_entity_groupablebyCSV_filename', help='Args file',
                           default='data/ontology/accounting_nl_entity_groupableby.csv',
                           type=str)

    argparser.add_argument('--nl_entity_filterablebyCSV_filename', help='Args file',
                           default='data/ontology/accounting_nl_entity_filterableby.csv',
                           type=str)

    argparser.add_argument('--RunningFolderPath', help='Args file',
                           default='.',
                           type=str)

    args = argparser.parse_args()

    model_api = NP1(model_args=args.model_args, NL_Entities_filename=args.NL_Entities_filename,
                    nl_entity_filterablebyCSV_filename=args.nl_entity_filterablebyCSV_filename,
                    nl_entity_groupablebyCSV_filename=args.nl_entity_groupablebyCSV_filename,
                    RunningFolderPath=args.RunningFolderPath,
                    use_cuda=args.use_cuda)

    print(model_api.greedy_translate(nl))

    resp = model_api.beam_translate(nl)
    print(resp)

    resp = model_api.reverse_translate(resp)
    print(resp)


if __name__ == '__main__':
    nl = "total sales from DUCKLING_TIME"
    main(nl)

