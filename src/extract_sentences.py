import pandas as pd
import nltk.data
from collections import defaultdict

from data.definition import RADIOLOGY_REPORTS
from data.definition import RADIOLOGY_SENTENCES_PER_REPORT


def extract_sentences():
    with open(RADIOLOGY_REPORTS, 'r') as fp:
        fp.seek(0)
        searchlines = fp.readlines()

        start_idx = 0 
        start_line = ""
        liver_num = 0

        final_reports_num = 0
        find_num = 0

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        results = defaultdict(list)

        for i, line in enumerate(searchlines):
            if "FINAL REPORT" in line:
                start_idx = i
                start_line = line
            if ("================================" in line or "(Over)" in line) and ("FINAL REPORT" in start_line):
                final_reports_num += 1
                all_findings_sentences = searchlines[start_idx+1:i]
                all_findings_sentences = " ".join(all_findings_sentences).replace("\n", "").strip()

                if "FINDINGS:" in all_findings_sentences:
                    find_num += 1

                if (" liver" in all_findings_sentences or " hepatic" in all_findings_sentences) and "FINDINGS:" in all_findings_sentences:
                    liver_num += 1
                    sentences = tokenizer.tokenize(all_findings_sentences)
             
                    for sent_idx, sent in enumerate(sentences):
                        if " liver" in sent or " hepatic" in sent:
                            results["first_sentence"].append(sentences[sent_idx])
                            results["second_sentence"].append("".join(sentences[sent_idx+1:sent_idx+2]))
                            results["third_sentence"].append("".join(sentences[sent_idx+2:sent_idx+3]))
                    # if "FIDNING" in indent_line or "TECHNIQUE:" in indent_line or "INDICATION" in indent_line or "HISTORY" in indent_line:

        print(f"Number of Liver/Hepatic reports: {liver_num}") # 73586 reports
        print(f"Number of final reports: {final_reports_num}")
        print(f"Number of finding: {find_num}")
        # df = pd.DataFrame(results)
        # df.to_csv(RADIOLOGY_SENTENCES_PER_REPORT, index=False)


if __name__ == '__main__':
    extract_sentences()