import pandas as pd
import nltk.data
from collections import defaultdict

from data.definition import RADIOLOGY_REPORTS
from data.definition import FINDINGS_RADIOLOGY_SENTENCES_PER_REPORT
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

        final_reports_num = 0
        find_liver_num = 0
        start_find_idx = 0

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

                if (" liver" in all_findings_sentences or " hepatic" in all_findings_sentences) and "FINDINGS:" in all_findings_sentences:
                    sentences = tokenizer.tokenize(all_findings_sentences)
             
                    for sent_idx, sent in enumerate(sentences):
                        if "FINDINGS:" in sent:
                            start_find_idx = sent_idx

                        if "IMPRESSION:" in sent:
                            for find_sent_idx, find_sent in enumerate(sentences[start_find_idx:sent_idx]):
                                if " liver" in find_sent or " hepatic" in find_sent:
                                    first_sent = sentences[start_find_idx:sent_idx][find_sent_idx]
                                    sec_sent = sentences[start_find_idx:sent_idx][find_sent_idx+1:find_sent_idx+2]
                                    third_sent = sentences[start_find_idx:sent_idx][find_sent_idx+2:find_sent_idx+3]

                                    first_sent = replace_leading_word(first_sent).strip()
                                 
                                    results["first_sentence"].append(filter_date(first_sent))
                                    results["second_sentence"].append(filter_date("".join(sec_sent)))
                                    results["third_sentence"].append(filter_date("".join(third_sent)))
                                    find_liver_num += 1

        print(f"Number of final reports: {final_reports_num}") # 606582 reports
        print(f"Number of finding: {find_liver_num}") # 23303 reports with liver in section FINDINGS
        
        df = pd.DataFrame(results)
        df.to_csv(FINDINGS_RADIOLOGY_SENTENCES_PER_REPORT, index=False)


if __name__ == '__main__':
    extract_sentences()