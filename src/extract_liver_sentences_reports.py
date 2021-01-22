import pandas as pd
import nltk.data
from collections import defaultdict

from data.definition import RADIOLOGY_REPORTS
from data.definition import RADIOLOGY_SENTENCES_PER_REPORT
from data.patterns import LIST_FINDINGS_WORDS


def filter_date(sent):
    if "[**" and "**]" in sent:
        start = sent.find('[**')
        end = sent.find('**]')

        if start != -1 and end != -1:
            sent = sent.replace(sent[start:end+3], "")
            
    return sent

def replace_leading_word(sent):
    for find_word in LIST_FINDINGS_WORDS:
        if find_word in sent:
            if "FINDINGS:" in sent and "CT OF THE ABDOMEN WITHOUT IV CONTRAST:" in sent:
                return sent.replace("FINDINGS:   CT OF THE ABDOMEN WITHOUT IV CONTRAST:", "")
            else:
                return sent.replace(find_word, "")
    return sent

def extract_sentences():
    with open(RADIOLOGY_REPORTS, 'r') as fp:
        fp.seek(0)
        searchlines = fp.readlines()

        start_idx = 0 
        start_line = ""
        liver_num = 0
        final_reports_num = 0

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

                if (" liver" in all_findings_sentences) or (" hepatic" in all_findings_sentences):
                    liver_num += 1
                    sentences = tokenizer.tokenize(all_findings_sentences)
             
                    for sent_idx, sent in enumerate(sentences):
                        if " liver" in sent or " hepatic" in sent:
                            first_sent = sentences[sent_idx]
                            sec_sent = sentences[sent_idx+1:sent_idx+2]
                            third_sent = sentences[sent_idx+2:sent_idx+3]

                            first_sent = replace_leading_word(first_sent).strip()

                            results["first_sentence"].append(first_sent)
                            results["second_sentence"].append("".join(sec_sent))
                            results["third_sentence"].append("".join(third_sent))

        print(f"Number of final reports: {final_reports_num}") # 606582 reports
        print(f"Number of Liver/Hepatic reports: {liver_num}") # 73586 reports

        df = pd.DataFrame(results)
        df.to_csv(RADIOLOGY_SENTENCES_PER_REPORT, index=False)


if __name__ == '__main__':
    extract_sentences()